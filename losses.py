class CosineSimilarityLoss(object):
    def __call__(self, a, b):
        return 1 - torch.sum(torch.mul(a, b) / (torch.norm(a, p=2) * torch.norm(b, p=2)))


class GazeAngularLoss(object):
    def __call__(self, gaze, gaze_hat):
        y = gaze.detach()
        y_hat = gaze_hat
        sim = F.cosine_similarity(y, y_hat, eps=1e-8)
        sim = F.hardtanh(sim, -1.0 + 1e-8, 1.0 - 1e-8)
        return torch.mean(torch.acos(sim) * (180 / np.pi))

      
class JS_DivergenseLoss(object):
    def __call__(self, a, b):
        loss = F.kl_div(F.log_softmax(a, dim=1), F.softmax(b, dim=1), reduction='batchmean') + \
               F.kl_div(F.log_softmax(b, dim=1), F.softmax(a, dim=1), reduction='batchmean')
        return loss
