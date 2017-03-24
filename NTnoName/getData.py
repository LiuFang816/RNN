# -*- coding:utf-8 -*-
#取出代码数据
def getData(fname1,fname2):
    f1=open(fname1)
    f2=open(fname2,'w')
    while 1:
        data=f1.readline()
        if not data:
            break
        if len(data.split('\t'))<4:
            continue
        data=data.split('\t')[3]
        f2.write(data+'\n')
    f1.close()
    f2.close()

import tokenize
#分解为token串,ENDMARKER分离代码段,数字用NUM表示
def token(fname1,fname2):
    f1=open(fname1)
    f2=open(fname2,'w')

    while 1:
        line=f1.readline()
        if not line:
            break
        line=line.replace(r'\n','\n')
        tokens=list(tokenize._my_tokenize(line,'utf-8'))

        for token in tokens:
            type=tokenize.tok_name[token.type]
            print('%-20s %-20r'%(tokenize.tok_name[token.type],token.string))
            if type =='STRING' and token.string[0:3]==('"""' or "'''"):
                continue
            if token.string=='\n':
                continue
            if type != 'COMMENT' and  type != 'NEWLINE' and type != 'NL' and type != 'ENCODING' and type!='ENDMARKER' and type!='NUMBER' :
                f2.write(token.string+' ')
            #todo 这里NUM会造成代码中出现好多NUM，给预测结果造成影响，后面和变量名一起处理
            if type == 'NUMBER' or type == 'ENDMARKER' :
                f2.write(type+' ')
    f1.close()
    f2.close()

# token('data/code/train.txt','data/code/trainNo#.txt')
# token('data/code/test.txt','data/code/testNo#.txt')
# token('data/code/valid.txt','data/code/validNo#.txt')


#todo 删除小于10（numsteps）个token的行,这里目前numsteps还不是灵活的
def getFinalData(fname1,fname2):
    f1=open(fname1)
    f2=open(fname2,'w')

    a=0
    data=f1.read().split('ENDMARKER')
    print(len(data))
    for i in range(len(data)):
        line=data[i]
        l=line.split(' ')
        if len(l)>=11:
            f2.write(line+'ENDMARKER ')
        else:
            a+=1
    print(a)
    f1.close()
    f2.close()

# getFinalData('data/code/trainNo#.txt','data/code/trainGT10.txt')
# getFinalData('data/code/testNo#.txt','data/code/testGT10.txt')
# getFinalData('data/code/validNo#.txt','data/code/validGT10.txt')









