import torch
import torch.nn as nn
from .backbone import SharedEncoder

class DinoHead(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class DinoModel(nn.Module):
    def __init__(self, out_dim=256, momentum=0.996):
        super().__init__()
        self.momentum = momentum

        # student
        self.student = SharedEncoder()
        self.student_head = DinoHead(self.student.out_dim, out_dim)

        # teacher
        self.teacher = SharedEncoder()
        self.teacher_head = DinoHead(self.teacher.out_dim, out_dim)
        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data = pt.data * self.momentum + ps.data * (1 - self.momentum)
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data = pt.data * self.momentum + ps.data * (1 - self.momentum)

    def forward(self, view1, view2):
        s_feat = self.student(view1)
        s_out = self.student_head(s_feat)

        with torch.no_grad():
            self.update_teacher()
            t_feat = self.teacher(view2)
            t_out = self.teacher_head(t_feat)

        return s_out, t_out
