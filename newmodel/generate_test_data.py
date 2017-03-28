# encoding:utf-8


stop_words={'{','}','ENDMARKER','NameConstant'}
stop_terminal_set={'AsName','ModuleName','ClassName','FuncName','AsyncFuncName'}
terminal_set= {'Num','Arg','Str','Bytes','True','False',
               'Name','AttrName','Key','UNK'}


def is_terminal(token):
    """
    处理终结符，终结符要么是terminal_set第一行中的特殊符号，或者是如Name('a')这种带括号的形式
    :param token:
    :param isTerminalSet:
    :return:
    """
    type=token[:token.index('(')] if token.endswith(')') else token
    if type in terminal_set:
        return True
    return False


def is_nonterminal(token):
    """
    处理终结符，终结符要么是terminal_set第一行中的特殊符号，或者是如Name('a')这种带括号的形式
    :param token:
    :param isTerminalSet:
    :return:
    """
    if token in terminal_set:
        return False
    if(token.endswith(')') and token.find(r'(')>=0):
        return False
    return True

def handle_line(line, wf, nums_steps=None,max_words_length=None,record_time_step=None,isTerminalSet=True):
    line=line.strip()
    cur_line_data=""
    count=0
    tokens=line.split(" ")
    if nums_steps and len(tokens)>nums_steps:
        return
    for token in tokens:
        count+=1
        if count==1:
            cur_line_data=token
        elif count>1:
            cur_line_data+=" "+token
        if(max_words_length and count>max_words_length):
            break
        if record_time_step is None or(count%record_time_step==0):
            if token in stop_words:
                continue
            if(is_terminal(token) and isTerminalSet and count>10):
                wf.write(cur_line_data+"\n")
            elif(is_nonterminal(token) and not isTerminalSet and count>10):
                wf.write(cur_line_data+"\n")


original_data_path=r'data/shuffle_train.txt'

is_terminal_set=False
save_path=r'data/train_terminal60.txt' if is_terminal_set else r'data/train_nonterminal60.txt'

# count=0
# with open(save_path,'w') as wf:
#     with open(original_data_path) as f:
#         for line in f:
#             handle_line(line, wf, 60, max_words_length=60,record_time_step=None,isTerminalSet=is_terminal_set)
#             count+=1
#             # if count>10:
#             #     break



