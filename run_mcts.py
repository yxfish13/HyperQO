import random
import sys
from ImportantConfig import Config
config = Config()
from sql2fea import TreeBuilder,value_extractor
from NET import TreeNet
from sql2fea import Sql2Vec
from TreeLSTM import SPINN

sys.stdout = open(config.log_file, "w")
random.seed(113)
with open(config.queries_file) as f:
    import json
    queries = json.load(f)

tree_builder = TreeBuilder()
sql2vec = Sql2Vec()
value_network = SPINN(head_num=config.head_num, input_size=7+2, hidden_size=config.hidden_size, table_num = 50,sql_size = 40*40+config.max_column).to(config.device)
for name, param in value_network.named_parameters():
    from torch.nn import init
    if len(param.shape)==2:
        init.xavier_normal(param)
    else:
        init.uniform(param)


net = TreeNet(tree_builder= tree_builder,value_network = value_network)
from Hinter import Hinter
from mcts import MCTSHinterSearch
mcts_searcher = MCTSHinterSearch()
hinter = Hinter(model = net,sql2vec = sql2vec,value_extractor = value_extractor,mcts_searcher = mcts_searcher)

print(len(queries))
for epoch in range(1):
    for idx,x in enumerate(queries[:5]):
        print('----',idx,'-----')

        pg_plan_time,pg_latency,mcts_time,hinter_plan_time,MPHE_time,hinter_latency,actual_plans,actual_time = hinter.hinterRun(x[0])
        print('pg plan:',pg_plan_time,'pg run:',pg_latency)
        print('mcts:',mcts_time,'plan gen:',hinter_plan_time,'MPHE:',MPHE_time,'hinter latency:',hinter_latency)
        print([actual_plans,actual_time])
        import json
        sys.stdout.flush()
        
    

    