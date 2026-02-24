import torch
import copy


class TeacherModel:
    def __init__(self, student_model, momentum=0.995):
        self.momentum = momentum
        self.teacher = copy.deepcopy(student_model)

        # Teacher never trains directly
        for p in self.teacher.parameters():
            p.requires_grad = False

    def update(self, student_model):
        """
        EMA weight update
        """
        with torch.no_grad():
            for t, s in zip(self.teacher.parameters(), student_model.parameters()):
                t.data = self.momentum * t.data + (1 - self.momentum) * s.data

    def forward(self, x):
        return self.teacher(x)