import torch
from torch.autograd import Variable
import torch.nn as nn
from qpth.qp import QPFunction
import cvxpy as cp
import numpy as np


def computeGramMatrix(A, B):
    #Constructs a linear kernel matrix between A and B 
    #for computing kernel similarity between two sets of vectors in SVMs

    assert(A.dim() == 3)
    assert(B.dim() == 3)
    assert(A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1,2)) #Batch matrix multiplication



def binv_safe(b_mat, use_pinv=True, rcond=1e-5):
    """
    Computes the inverse of each matrix in a batch.
    Falls back to pseudoinverse if use_pinv is True and inversion fails.
    
    Parameters:
      b_mat: Tensor of shape (batch_size, n, n)
      use_pinv: If True, use pseudoinverse when regular inverse fails
      rcond: Cutoff for small singular values in pseudoinverse
    Returns:
      Tensor of shape (batch_size, n, n)
    """
    b_mat = b_mat.cuda()
    try:
        batch_size, n, _ = b_mat.shape
        I = torch.eye(n, device=b_mat.device).expand(batch_size, n, n)
        return torch.linalg.solve(b_mat, I)
    except RuntimeError as e:
        if not use_pinv:
            raise e
        return torch.linalg.pinv(b_mat, rcond=rcond)

def one_hot(indices, depth):
    """
    Converts integer class indices to one-hot encodings.

    Parameters:
        indices: LongTensor of shape (...), values in [0, depth-1]
        depth: int, number of classes
    Returns:
        One-hot encoded tensor of shape (..., depth)
    """
    indices = indices.cuda()
    shape = indices.shape + (depth,)
    one_hot = torch.zeros(shape, device=indices.device, dtype=torch.float)
    return one_hot.scatter_(-1, indices.unsqueeze(-1), 1)


def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape([matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))


def ProtoNetHead(query, support, support_labels, n_way, n_shot, normalize=True):
    tasks_per_batch, n_query, d = query.shape
    _, n_support, _ = support.shape

    assert n_support == n_way * n_shot

    support_labels_one_hot = one_hot(support_labels.view(-1), n_way).view(tasks_per_batch, n_support, n_way).transpose(1, 2)
    prototypes = torch.bmm(support_labels_one_hot, support)
    prototypes = prototypes / support_labels_one_hot.sum(dim=2, keepdim=True)

    # Efficient L2 distance computation
    AA = query.pow(2).sum(dim=2, keepdim=True)
    BB = prototypes.pow(2).sum(dim=2).unsqueeze(1)
    AB = torch.bmm(query, prototypes.transpose(1, 2))
    logits = - (AA - 2 * AB + BB)

    if normalize:
        logits = logits / d

    return logits




def MetaOptNetHead_SVM_He(query, support, support_labels, n_way, n_shot, C_reg=0.01):
    tasks_per_batch, n_support, d = support.shape
    n_query = query.size(1)
    device = support.device
    dtype = torch.float32
    logits_all = []

    for t in range(tasks_per_batch):
        query_t = query[t]         # (n_query, d)
        support_t = support[t]     # (n_support, d)
        labels_t = support_labels[t]  # (n_support,)

        # Normalize support and query for numerical stability
        support_t = support_t / support_t.norm(dim=1, keepdim=True)
        query_t = query_t / query_t.norm(dim=1, keepdim=True)

        K = support_t @ support_t.T  # (n_support, n_support)
        V = (labels_t * n_way - 1.0) / (n_way - 1.0)
        V = V.unsqueeze(1).expand(-1, n_way)  # (n_support, n_way)
        V_expected = torch.arange(n_way, device=device).float()
        V = (V == V_expected).float()

        G = K * (V @ V.T)
        G += torch.eye(n_support, device=device) * 1e-6  # Ensure G is PSD

        # Convert G to NumPy and wrap it as PSD
        G_np = G.cpu().detach().numpy()
        e_np = -np.ones(n_support, dtype=np.float64)

        z = cp.Variable(n_support)
        objective = cp.Minimize(0.5 * cp.quad_form(z, G_np) + e_np @ z)
        constraints = [z >= 0, z <= C_reg]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5, verbose=False)

        if z.value is None:
            raise RuntimeError(f"Solver failed at task {t} with status {prob.status}")
        
        z_opt = torch.tensor(z.value, dtype=dtype, device=device)

        compat = query_t @ support_t.T  # (n_query, n_support)
        scores = (compat * z_opt.unsqueeze(0)).view(n_query, n_shot, n_way).sum(1)
        logits_all.append(scores)

    return torch.stack(logits_all, dim=0)  # (tasks_per_batch, n_query, n_way)
def MetaOptNetHead_SVM_CS(query, support, support_labels, n_way, n_shot, C_reg=0.1, double_precision=False, maxIter=15):
    device = query.device
    dtype = torch.float64 if double_precision else torch.float32

    tasks_per_batch, n_query, d = query.shape
    n_support = support.size(1)

    assert n_support == n_way * n_shot

    K = computeGramMatrix(support, support)  # (T, n_support, n_support)

    eye_way = torch.eye(n_way, device=device, dtype=dtype)
    block_K = batched_kronecker(K, eye_way)  # (T, n_way*n_support, n_way*n_support)
    block_K += 1e-4 * torch.eye(n_way * n_support, device=device, dtype=dtype).unsqueeze(0)

    labels_oh = one_hot(support_labels.view(-1), n_way).view(tasks_per_batch, n_support, n_way).reshape(tasks_per_batch, -1)
    e = -labels_oh.to(dtype)

    C = torch.eye(n_way * n_support, device=device, dtype=dtype).expand(tasks_per_batch, -1, -1)
    h = (C_reg * labels_oh).to(dtype)

    ones = torch.ones((tasks_per_batch, 1, n_way), device=device, dtype=dtype)
    eye_supp = torch.eye(n_support, device=device, dtype=dtype).expand(tasks_per_batch, -1, -1)
    A = batched_kronecker(eye_supp, ones)
    b = torch.zeros((tasks_per_batch, n_support), device=device, dtype=dtype)

    Q = block_K.to(dtype)
    qp_sol = QPFunction(verbose=False, maxIter=maxIter)(Q, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())

    K_qs = computeGramMatrix(support, query).to(dtype)
    K_qs = K_qs.unsqueeze(3).expand(-1, -1, -1, n_way)

    alpha = qp_sol.view(tasks_per_batch, n_support, n_way).to(dtype)
    alpha_exp = alpha.unsqueeze(2).expand(-1, -1, n_query, -1)

    logits = torch.sum(alpha_exp * K_qs, dim=1)
    return logits
class ClassificationHead(nn.Module):
    def __init__(self, base_learner='MetaOptNet', enable_scale=True):
        super(ClassificationHead, self).__init__()
        
        if ('Proto' in base_learner):
            self.head = ProtoNetHead
        elif ('SVM-He' in base_learner):
            self.head = MetaOptNetHead_SVM_He
        elif ('SVM-CS' in base_learner):
            self.head = MetaOptNetHead_SVM_CS
        else:
            print ("Cannot recognize the base learner type")
            assert(False)
        
        # Add a learnable scale
        self.enable_scale = enable_scale
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))
        
    def forward(self, query, support, support_labels, n_way, n_shot, **kwargs):
        if self.enable_scale:
            return self.scale * self.head(query, support, support_labels, n_way, n_shot, **kwargs)
        else:
            return self.head(query, support, support_labels, n_way, n_shot, **kwargs)
