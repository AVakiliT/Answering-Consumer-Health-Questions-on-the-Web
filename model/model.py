from pytorch_lightning import LightningModule


class s22(LightningModule):
    def __init__(self):
        super().__init__()
        self.qe = None
        self.ce = None
        self.qa = None


        def forward(q_bkl,p_bkl, qp_bkl):
            self.qe()