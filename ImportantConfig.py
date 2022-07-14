import torch
from math import log
class Config:
    def __init__(self,):
        self.datafile = 'JOBqueries.workload'
        self.schemaFile = "schema.sql"
        self.database = 'imdb'
        self.user = 'postgres'
        self.password = ""
        self.dataset = 'JOB'
        self.userName = self.user
        self.usegpu = True
        self.head_num = 10
        self.input_size = 9
        self.hidden_size = 128
        self.batch_size = 256
        self.ip = "127.0.0.1"
        self.port = 5432
        self.device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
        self.cpudevice = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
        self.var_weight = 0.00 #for au, 0:disable,0.01:enable
        self.max_column = 100
        self.max_alias_num  = 40
        self.cost_test_for_debug = True
        self.max_hint_num = 20
        self.max_time_out = 120*1000
        self.threshold = log(2)/log(self.max_time_out)
        self.leading_length = 2
        self.try_hint_num = 4
        self.mem_size = 2000
        self.mcts_v = 1.1
        self.mcts_input_size = self.max_alias_num*self.max_alias_num+self.max_column
        self.searchFactor = 4
        self.U_factor = 0.0
        self.log_file = 'log.txt'
        self.latency_file = 'latency_record.txt'
        self.queries_file = 'workload/JOB_static.json'
        self.id2aliasname = {0: 'start', 1: 'chn', 2: 'ci', 3: 'cn', 4: 'ct', 5: 'mc', 6: 'rt', 7: 't', 8: 'k', 9: 'lt', 10: 'mk', 11: 'ml', 12: 'it1', 13: 'it2', 14: 'mi', 15: 'mi_idx', 16: 'it', 17: 'kt', 18: 'miidx', 19: 'at', 20: 'an', 21: 'n', 22: 'cc', 23: 'cct1', 24: 'cct2', 25: 'it3', 26: 'pi', 27: 't1', 28: 't2', 29: 'cn1', 30: 'cn2', 31: 'kt1', 32: 'kt2', 33: 'mc1', 34: 'mc2', 35: 'mi_idx1', 36: 'mi_idx2', 37: 'an1', 38: 'n1', 39: 'a1'}
        self.aliasname2id = {'kt1': 31, 'chn': 1, 'cn1': 29, 'mi_idx2': 36, 'cct1': 23, 'n': 21, 'a1': 39, 'kt2': 32, 'miidx': 18, 'it': 16, 'mi_idx1': 35, 'kt': 17, 'lt': 9, 'ci': 2, 't': 7, 'k': 8, 'start': 0, 'ml': 11, 'ct': 4, 't2': 28, 'rt': 6, 'it2': 13, 'an1': 37, 'at': 19, 'mc2': 34, 'pi': 26, 'mc': 5, 'mi_idx': 15, 'n1': 38, 'cn2': 30, 'mi': 14, 'it1': 12, 'cc': 22, 'cct2': 24, 'an': 20, 'mk': 10, 'cn': 3, 'it3': 25, 't1': 27, 'mc1': 33}
        self.modelpath = 'model/'
        self.offset = 20

