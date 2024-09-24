from lambeq import BobcatParser, AtomicType, RemoveCupsRewriter, UnifyCodomainRewriter, Rewriter, IQPAnsatz

remove_cups = RemoveCupsRewriter()

parser = BobcatParser()
rewriter = Rewriter(['auxiliary',
                     'connector',
                     'coordination',
                     'determiner',
                     'object_rel_pronoun',
                     'subject_rel_pronoun',
                     'postadverb',
                     'preadverb',
                     'prepositional_phrase'])

N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = AtomicType.PREPOSITIONAL_PHRASE

ansatz = IQPAnsatz({N: 1, S: 1, P:1}, n_layers=1, n_single_qubit_params=3)

def sent2dig(sentence, pro, ref):
    diagram = parser.sentence2diagram(sentence)
    diagram = diagram >> (S, 2, 1)
    pro_box_idx = next(i for i, box in enumerate(diagram.boxes) if box.name.casefold() == pro.casefold())
    ref_box_idx = next(i for i, box in enumerate(diagram.boxes) if box.name.casefold() == ref.casefold())
    diagram = rewriter(remove_cups(diagram)).normal_form()
    return diagram
