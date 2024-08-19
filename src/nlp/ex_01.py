# Trabalho feito pelos alunos:
# Bruno Ribeiro Medeiros
# Pedro

# Similaridade sintática
# Distancias: [Hammiing | Levenshtein (Edit Distance)]

# S = 2C / (A + B)
#     A digramas unicos na primeira palavra
#     B digramas unicos na segunda palavra
#     C digramas unicos compartilhados

def get_unique_grams(grama_list):
    return list(set(grama_list))


# N-Gram
if __name__ == '__main__':
    # lexico = [abacate, abacaxi, abobora, abobrinha, ananás, maça, mamão, manga, melancia, melão, mexerica, morango]
    lexico = ["abacate", "abacaxi", "abobora", "abobrinha", "ananás", "maça", "mamão", "manga", "melancia", "melão",
              "mexerica", "morango"]

    # lexico = ["abacate"]
    #
    # dictionary = {}
    # for word in lexico:
    #     if word not in dictionary:
    #         dictionary[word] = []
    #
    #     for i, char in enumerate(word):
    #         dictionary[word].append(word)

    d = {"a": 1, "b":2}
    for dic in d:
        print(dic)










