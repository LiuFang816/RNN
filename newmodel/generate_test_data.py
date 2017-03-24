# encoding:utf-8


stop_words={'{','}','NameConstant'}
terminal_set= {'Num','Arg','Str','Bytes','True','False',
               'Name','AsName','AttrName','Key','ModuleName','ClassName','FuncName','AsyncFuncName'}


def is_terminal(type):
    """
    处理终结符，终结符要么是terminal_set第一行中的特殊符号，或者是如Name('a')这种带括号的形式
    :param type:
    :param isTerminalSet:
    :return:
    """
    if type in terminal_set:
        return True
    if(type.endswith(')') and type.find(r'(')>=0):
        return True
    return False


def is_nonterminal(type):
    """
    处理终结符，终结符要么是terminal_set第一行中的特殊符号，或者是如Name('a')这种带括号的形式
    :param type:
    :param isTerminalSet:
    :return:
    """
    if type in terminal_set:
        return False
    if(type.endswith(')') and type.find(r'(')>=0):
        return False
    return True

def handle_line(line, wf, isTerminalSet=True):
    line=line.strip()
    cur_line_data=""
    count=0
    types=line.split(" ")
    for type in types:
        count+=1
        cur_line_data+=" "+type
        if count%100==0:
            if type in stop_words:
                continue
            if(is_terminal(type) and isTerminalSet and count>10):
                wf.write(cur_line_data+"\n")
            elif(is_nonterminal(type) and not isTerminalSet and count>10):
                wf.write(cur_line_data+"\n")


original_data_path=r'D:\MyProjectsRepertory\Python_project\s-lstm\data\test.txt'

is_terminal_set=True
save_path=r'test_terminal.txt' if is_terminal_set else r'test_nonterminal.txt'

count=0
with open(save_path,'w') as wf:
    with open(original_data_path) as f:
        for line in f:
            handle_line(line, wf, isTerminalSet=is_terminal_set)
            count+=1
            # if count>10:
            #     break



