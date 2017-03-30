import os
import pickle
from nltk.corpus import brown
from nltk.tokenize import word_tokenize


def calc_prob(tagged_paras_list):
    # Get States
    states = set()
    for para in tagged_paras_list:
        for sent in para:
            for _, tag in sent:
                states.add(tag)
    states = list(states)
    start_p = {}
    trans_p = {}
    emit_p = {}

    for para in tagged_paras_list:
        for sent in para:
            # empty previous_tag means start of the sentences
            previous_tag = ""
            for word, tag in sent:
                word = word.lower()
                try:
                    start_p[tag] += 1
                except KeyError:
                    start_p[tag] = 1

                if previous_tag != "":
                    if tag not in trans_p:
                        trans_p[tag] = {}
                    try:
                        trans_p[tag][previous_tag] += 1
                    except KeyError:
                        trans_p[tag][previous_tag] = 1
                previous_tag = tag

                if tag not in emit_p:
                    emit_p[tag] = {}
                try:
                    emit_p[tag][word] += 1
                except KeyError:
                    emit_p[tag][word] = 1

    total = sum(start_p.values())
    for tag in start_p:
        start_p[tag] /= total

    for tag1 in trans_p:
        total = sum(trans_p[tag1].values())
        for tag2 in trans_p[tag1]:
            trans_p[tag1][tag2] /= total

    for tag in trans_p:
        total = sum(emit_p[tag].values())
        for word in emit_p[tag]:
            emit_p[tag][word] /= total

    return tuple(states), start_p, trans_p, emit_p


def pos_tagger(sentence, path="probabilities.pickle"):
    if not os.path.exists(path):
        data = calc_prob(brown.tagged_paras())
        pickle.dump(data, open(path, "wb"))
    else:
        data = pickle.load(open(path, "rb"))

    states, start_p, trans_p, emit_p = data

    return viterbi(sentence,
                  states,
                  start_p,
                  trans_p,
                  emit_p)


def viterbi(sentence, states, start_p, trans_p, emit_p):
    obs = [tok.lower() for tok in word_tokenize(sentence)]
    V = [{}]
    for st in states:
        try:
            emit_probability = emit_p[st][obs[0]]
        except KeyError:
            emit_probability = 0
        V[0][st] = {"prob": start_p[st] * emit_probability, "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_helper_list = []
            for prev_st in states:
                try:
                    trans_probability = trans_p[prev_st][st]
                    pre = V[t - 1][prev_st]["prob"]
                except KeyError:
                    trans_probability = 0
                    pre = 0
                max_helper_list.append(pre * trans_probability)
            max_tr_prob = max(max_helper_list)
            for prev_st in states:
                try:
                    if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                        max_prob = max_tr_prob * emit_p[st][obs[t]]
                        V[t][st] = {"prob": max_prob, "prev": prev_st}
                        break
                except KeyError:
                    pass

    opt = []
    # The highest probability
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    return opt, max_prob


def main():
    import sys
    if len(sys.argv) == 2:
        print(pos_tagger(sys.argv[1]))
    else:
        print(pos_tagger("A smooth sea never made a skilled sailor"))


if __name__ == "__main__":
    main()
