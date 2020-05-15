import math
import numpy as np 
from config.config import *
EPS = 1e-8


class MCTS():
    """
    This class is for the MCTS search
    """
    def __init__(self, game, avp_net):
        self.game = game
        self.avp_net = avp_net
        self.Qsa = {}         # store Q values for (s, a)
        self.Nsa = {}         # store visited times for edge s,a
        self.Ns = {}          # store visited times for state s
        self.Ps = {}          # store initial policy for state s returned by avp network
        
        self.Es = {}          # stores game's end state
    
    def get_action_prob(self, img, s, tmp=1):

        for i in range(FLAGS.num_mcts_sim):
            level = 0
            self.search(s, level, img)
        counts = [self.Nsa[((s, 0), a)] if ((s, 0),  a) in self.Nsa else 0 for a in range(self.game.action_num)]
        if tmp == 0:
            best_act = np.argmax(counts)
            probs = [0] * len(counts)
            probs[best_act] = 1
            return probs
        
        counts = [x**(1./tmp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs

    def search(self, s, level, img):
        if (s, level) not in self.Es:
            self.Es[(s, level)] = self.game.get_game_ended(level, img)
        if self.Es[(s, level)] != 0:
            # terminal node 
            return self.Es[(s, level)]
        
        if (s, level) not in self.Ps:
            # leaf node(expand and evaluate)
            self.Ps[(s, level)], v = self.avp_net.predict(img)
            self.Ns[(s, level)] = 0
            return v
        
        cur_best = -float('inf')
        best_act = -1

        # select the best action according to the PUCT equation
        for a in range(self.game.action_num):
            if ((s, level), a) in self.Qsa:
                r = self.Qsa[((s, level), a)] + FLAGS.cpuct*self.Ps[(s, level)][a] * \
                    math.sqrt(self.Ns[(s, level)])/(1+self.Nsa[((s, level), a)])
            else:
                r = FLAGS.cpuct*self.Ps[(s, level)][a]*math.sqrt(self.Ns[(s, level)]+EPS)
            if r > cur_best:
                best_act = a
                cur_best = r
        a = best_act
        next_img = self.game.get_next_state(img, a)
        
        v = self.search(a, level+1, next_img)

        if ((s, level), a) in self.Qsa:
            self.Qsa[((s, level), a)] = (self.Nsa[((s, level), a)]*self.Qsa[((s, level), a)] + v) / \
                                        (self.Nsa[((s, level), a)]+1)
            self.Nsa[((s, level), a)] += 1
        else:
            self.Qsa[((s, level), a)] = v
            self.Nsa[((s, level), a)] = 1

        self.Ns[(s, level)] += 1
        return v
