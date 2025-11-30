import torch
import torch.nn as nn
import torch.nn.functional as F

class DinoLoss(nn.Module):
    def __init__(self, temp_student=0.1, temp_teacher=0.04, center_m=0.9, out_dim=256):
        super().__init__()
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher
        self.center_m = center_m
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_out):
        self.center = self.center * self.center_m + teacher_out.mean(dim=0, keepdim=True) * (1 - self.center_m)

    def forward(self, student, teacher):
        teacher = F.softmax((teacher - self.center) / self.temp_teacher, dim=-1)
        student = F.log_softmax(student / self.temp_student, dim=-1)

        loss = -(teacher * student).sum(dim=1).mean()

        self.update_center(teacher)
        return loss
