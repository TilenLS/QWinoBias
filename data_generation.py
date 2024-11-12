from lambeq import BobcatParser, AtomicType, RemoveCupsRewriter, UnifyCodomainRewriter, Rewriter, IQPAnsatz
from discopro.grammar import tensor
from lambeq.backend.grammar import Spider
from discopro.anaphora import connect_anaphora_on_top
import pandas as pd
import pickle

path = 'wino/data/data_final/'

remove_cups = RemoveCupsRewriter()

parser = BobcatParser()
rewriter = Rewriter(['curry'])

N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = AtomicType.PREPOSITIONAL_PHRASE

def sent2dig(sentence, pro1, pro2, ref):
    diagram = parser.sentence2diagram(sentence)
    # diagram2 = parser.sentence2diagram(sent1)
    # diagram = tensor(diagram1, diagram2)
    # diagram = diagram >> Spider(S, 2, 1)
    pro_box_idx = next(i for i, box in enumerate(diagram.boxes) if 
                       (box.name.casefold() == pro1.casefold() or box.name.casefold() == pro2.casefold()))
    ref_box_idx = next(i for i, box in enumerate(diagram.boxes) if box.name.casefold() == ref.casefold())
    diagram = connect_anaphora_on_top(diagram, pro_box_idx, ref_box_idx)
    diagram = rewriter(remove_cups(diagram)).normal_form()
    return diagram

def gen_data(path, file, folder):
    df = pd.read_csv(path + '/' + file + '.csv', index_col=0)
    df = df.sample(frac=1)
    
    circuits, labels, diagrams, sentences = [],[],[],[]
    
    for i, row in tqdm(df.iterrows(), total=len(df), position=0, leave=True):
        col = random.choice(['referent', 'wrong_referent'])
        sent, pro1, pro2, right_ref, wrong_ref = row[['Sentence', 'Pronoun 1', 'Pronoun 2', 'Right Referent', 'Wrong Referent']]
        try:
            diag_right = sent2dig(sent.strip(), pro1.strip(), pro2.strip(), ref.strip())
            diag_wrong = sent2dig(sent.strip(), pro1.strip(), pro2.strip(), ref.strip())
            diagrams.append(diag_right)
            diagrams.append(diag_wrong)
            circuits.append(ansatz(diag_right))
            circuits.append(ansatz(diag_wrong))
            labels.append([0,1])
            labels.append([1,0])
            sentences.append(sent)
            sentences.append(sent)
        except Exception as err:
            tqdm.write(f"Error: {err}".strip(), file=sys.stderr)

    if not os.path.exists(os.getcwd()+'/data/'+folder):
        os.mkdir(os.getcwd()+folder)
    
    f = open('data/11113/'+file+'.pkl', 'wb')
    pickle.dump(list(zip(circuits, labels, diagrams, sentences)), f)
    f.close()

def set_params(noun_q=1, sent_q=1, pp_q=1, n_layers=1, n_rots=3):
    
    global ansatz = IQPAnsatz({N: noun_q, S: sent_q, P:pp_q}, 
                              n_layers=n_layers, 
                              n_single_qubit_params=n_rots)

# set_params()
# gen_data(path, 'test')
# gen_data(path, 'train')
# gen_data(path, 'val')
# gen_data(path, 'unseen')