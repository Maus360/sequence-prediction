import numpy as np

# 1 2 3 4 5 6 7


def init(
    sequence: list,
    p: int,
    error: int,
    max_iter: int,
    code_for_learning: str,
    code_for_training: str,
):
    if sequence == None:
        sequence = list(
            map(int, input("Enter sequence, split numbers by space:\n").split())
        )
    q = len(sequence)
    if p == None:
        p = int(input("Enter window size:\n"))
    if error == None:
        error = int(input("Enter max learning error:\n"))
    if max_iter == None:
        max_iter = int(input("Enter max number of iterations:\n"))
    if code_for_learning == None:
        code_for_learning = input(
            "Enter learning code:\n"
        )  # on\off for first|on\off for others
    if code_for_training == None:
        code_for_training = input(
            "Enter training code:\n"
        )  # on\off for first|on\off for others
    if p > q:
        raise ("Invalid size of window, must be less then q")

    x = []
    y = []
    i = 0
    while i + p < q:
        x.append(sequence[i : i + p])
        y.append(sequence[i + p])
        i += 1
    y = np.array(y)
    x = np.array(x)
    run(x, y, p, q, error, max_iter, code_for_learning, code_for_training)
    return 0


def run(
    x: np.array,
    y: np.array,
    p: int,
    q: int,
    error: int,
    max_iter: int,
    code_for_learning: str,
    code_for_training: str,
):
    m = 4
    alpha = 0.000_000_000_05
    error_all = 0
    context = np.zeros((x.shape[0], m))
    x = np.concatenate((x, context), axis=1)
    w1 = np.random.rand(p + m, m) * 2 - 1
    w2 = np.random.rand(m, 1) * 2 - 1

    for j in range(max_iter):
        for i in range(x.shape[0]):
            hidden_layer = np.matmul(x[i], w1)
            output = np.matmul(hidden_layer, w2)
            dy = output - y[i]
            w1 -= alpha * dy * np.matmul(np.array([x[i]]).transpose(), w2.transpose())
            w2 -= alpha * dy * np.array([hidden_layer]).transpose()
            try:
                x[i + 1][-m:] = hidden_layer
            except:
                x[0][-m:] = hidden_layer
        for i in range(x.shape[0]):
            hidden_layer = np.matmul(x[i], w1)
            output = np.matmul(hidden_layer, w2)
            dy = output - y[i]
            error_all += (dy ** 2).sum()

        print(j, " ", error_all)


if __name__ == "__main__":
    print(
        init(
            sequence=None,
            p=None,
            error=100,
            max_iter=100_000_000_000,
            code_for_learning="00",
            code_for_training="00",
        )
    )

