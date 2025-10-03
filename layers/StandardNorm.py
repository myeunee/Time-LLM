import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        
        # 분산이 0이거나 NaN인 경우를 방지하기 위한 안전 조치
        var = torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)
        # 분산이 너무 작은 경우 최소값 설정
        var = torch.clamp(var, min=1e-8)
        # NaN 값을 1e-8로 대체
        var = torch.nan_to_num(var, nan=1e-8)
        
        self.stdev = torch.sqrt(var + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        try:
            if self.subtract_last:
                x = x - self.last
            else:
                x = x - self.mean
                
            # 0으로 나누는 것을 방지
            x = x / (self.stdev + 1e-8)
            
            # NaN 값 처리
            x = torch.nan_to_num(x, nan=0.0)
            
            if self.affine:
                x = x * self.affine_weight
                x = x + self.affine_bias
            return x
        except Exception as e:
            print(f"정규화 중 오류 발생: {e}")
            return x  # 오류 발생 시 원본 반환

    def _denormalize(self, x):
        if self.non_norm:
            return x
        try:
            if self.affine:
                x = x - self.affine_bias
                x = x / (self.affine_weight + self.eps * self.eps)
            
            # NaN 값 처리
            x = torch.nan_to_num(x, nan=0.0)
            
            x = x * self.stdev
            if self.subtract_last:
                x = x + self.last
            else:
                x = x + self.mean
            
            # 최종 결과에서 NaN 값 처리
            x = torch.nan_to_num(x, nan=0.0)
            return x
        except Exception as e:
            print(f"역정규화 중 오류 발생: {e}")
            return x  # 오류 발생 시 원본 반환
