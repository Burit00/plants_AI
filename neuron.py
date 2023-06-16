import hickle as hkl
import numpy as np
import nnet as net


class mlp_ma_3w:
    def __init__(self, x, y_t, K1, K2, lr, err_goal, disp_freq, mc, ksi_inc, ksi_dec, er, max_epoch):
        self.x = x
        self.L = self.x.shape[0]
        self.y_t = y_t
        self.K1 = K1
        self.K2 = K2
        self.lr = lr
        self.err_goal = err_goal
        self.disp_freq = disp_freq
        self.mc = mc
        self.ksi_inc = ksi_inc
        self.ksi_dec = ksi_dec
        self.er = er
        self.max_epoch = max_epoch
        self.K3 = y_t.shape[0]
        self.SSE_vec = []
        self.PK_vec = []
        self.w1, self.b1 = net.nwtan(self.K1, self.L)
        self.w2, self.b2 = net.rands(self.K2, self.K1)
        self.w3, self.b3 = net.rands(self.K3, self.K2)
        # self.w1, self.b1 = net.nwtan(self.K1, self.L)
        # self.w2, self.b2 = net.nwtan(self.K2, self.K1)
        # self.w3, self.b3 = net.rands(self.K3, self.K2)
        # hkl.dump([self.w1, self.b1, self.w2, self.b2, self.w3, self.b3], 'wagi3w.hkl')
        # self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = hkl.load('wagi3w.hkl')
        self.w1_t_1, self.b1_t_1, self.w2_t_1, self.b2_t_1, self.w3_t_1, self.b3_t_1 = \
            self.w1, self.b1, self.w2, self.b2, self.w3, self.b3
        self.SSE = 0
        self.lr_vec = list()

    def predict(self, x):
        self.y1 = net.tansig(np.dot(self.w1, x), self.b1)
        self.y2 = net.tansig(np.dot(self.w2, self.y1), self.b2)
        self.y3 = net.purelin(np.dot(self.w3, self.y2), self.b3)
        return self.y3

    def train(self, x_train, y_train):
        for epoch in range(1, self.max_epoch + 1):
            self.y3 = self.predict(x_train)
            self.e = y_train - self.y3

            self.SSE_t_1 = self.SSE
            self.SSE = net.sumsqr(self.e)
            self.PK = (1 - sum((abs(self.e) >= 0.5).astype(int)[0]) / self.e.shape[1]) * 100
            self.PK_vec.append(self.PK)
            if self.SSE < self.err_goal or self.PK == 100:
                break

            if np.isnan(self.SSE):
                break
            else:
                if self.SSE > self.er * self.SSE_t_1:
                    self.lr *= self.ksi_dec
                elif self.SSE < self.SSE_t_1:
                    self.lr *= self.ksi_inc
            self.lr_vec.append(self.lr)

            self.d3 = net.deltalin(self.y3, self.e)
            self.d2 = net.deltatan(self.y2, self.d3, self.w3)
            self.d1 = net.deltatan(self.y1, self.d2, self.w2)
            self.dw1, self.db1 = net.learnbp(self.x, self.d1, self.lr)
            self.dw2, self.db2 = net.learnbp(self.y1, self.d2, self.lr)
            self.dw3, self.db3 = net.learnbp(self.y2, self.d3, self.lr)

            self.w1_temp, self.b1_temp, self.w2_temp, self.b2_temp, self.w3_temp, self.b3_temp = \
                self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy(), self.w3.copy(), self.b3.copy()

            self.w1 += self.dw1 + self.mc * (self.w1 - self.w1_t_1)
            self.b1 += self.db1 + self.mc * (self.b1 - self.b1_t_1)
            self.w2 += self.dw2 + self.mc * (self.w2 - self.w2_t_1)
            self.b2 += self.db2 + self.mc * (self.b2 - self.b2_t_1)
            self.w3 += self.dw3 + self.mc * (self.w3 - self.w3_t_1)
            self.b3 += self.db3 + self.mc * (self.b3 - self.b3_t_1)
            self.w1_t_1, self.b1_t_1, self.w2_t_1, self.b2_t_1, self.w3_t_1, self.b3_t_1 = \
                self.w1_temp, self.b1_temp, self.w2_temp, self.b2_temp, self.w3_temp, self.b3_temp

            self.SSE_vec.append(self.SSE)
            if epoch % self.disp_freq == 0:
                print("Epoch: {} | SSE: {}".format(epoch, self.SSE))
