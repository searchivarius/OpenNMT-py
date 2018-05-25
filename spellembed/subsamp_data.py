import numpy as np

SUB_SHARE = 0.5

TRAIN_DE = '../data/MTTT/tok/ted_train_en-de.tok.clean.de.txt'
TRAIN_EN = '../data/MTTT/tok/ted_train_en-de.tok.clean.en.txt'
DEV_DE = '../data/MTTT/tok/ted_dev_en-de.tok.de.txt'
DEV_EN = '../data/MTTT/tok/ted_dev_en-de.tok.en.txt'
TEST_DE = '../data/MTTT/tok/ted_test1_en-de.tok.de.txt'
TEST_EN = '../data/MTTT/tok/ted_test1_en-de.tok.en.txt'

SUB_TRAIN_DE = '../data/MTTT/tok/sub_ted_train_en-de.tok.clean.de.txt'
SUB_TRAIN_EN = '../data/MTTT/tok/sub_ted_train_en-de.tok.clean.en.txt'
SUB_DEV_DE = '../data/MTTT/tok/sub_ted_dev_en-de.tok.de.txt'
SUB_DEV_EN = '../data/MTTT/tok/sub_ted_dev_en-de.tok.en.txt'
SUB_TEST_DE = '../data/MTTT/tok/sub_ted_test1_en-de.tok.de.txt'
SUB_TEST_EN = '../data/MTTT/tok/sub_ted_test1_en-de.tok.en.txt'

np.random.seed(0)

#Split func

def split(src_de,src_en,dest_de,dest_en,share=SUB_SHARE):

    print(src_de)
    print(src_en)

    with open(src_de) as f:
        lines_de = f.readlines()

    with open(src_en) as f:
        lines_en = f.readlines()

    N = len(lines_de)
    assert(N==len(lines_en))
    print(N)
    print(share)

    lens = []
    for i in range(N):
        lens.append(0.5*len(lines_de[i])+0.5*len(lines_en[i]))

    lens = np.array(lens)
    pct = np.percentile(lens,[0,25,50,75,100])
    print(pct)

    # obtain indices per quantile
    idxs_per_quantile = {j: [] for j in range(4)}

    for i in range(N):
        for j in range(len(idxs_per_quantile)):
            #print(j,pct[j],pct[j+1])
            if lens[i] >= pct[j] and lens[i] < pct[j+1]+0.5:
                idxs_per_quantile[j].append(i)

    #print(idxs_per_quantile)
    sub_idx = []

    for j in range(4):
        np.random.shuffle(idxs_per_quantile[j])
        subN = int(share * len(idxs_per_quantile[j]))
        print(subN)
        sub_idx.extend(idxs_per_quantile[j][:subN])

    print(len(sub_idx))

    with open(dest_de, "w") as f:
        for i in range(N):
            if i in sub_idx:
                f.write(lines_de[i])

    with open(dest_en, "w") as f:
        for i in range(N):
            if i in sub_idx:
                f.write(lines_en[i])


# Process files
split(TRAIN_DE,TRAIN_EN,SUB_TRAIN_DE,SUB_TRAIN_EN)
split(DEV_DE,DEV_EN,SUB_DEV_DE,SUB_DEV_EN)
split(TEST_DE,TEST_EN,SUB_TEST_DE,SUB_TEST_EN)