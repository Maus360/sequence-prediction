import numpy as np


def start(
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
    return run(x, y, p, q, error, max_iter, code_for_learning, code_for_training)


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
    m = 2
    alpha = 0.000000005
    error_all = 0
    k = 0
    if code_for_learning[0] == "1":
        context = np.zeros((x.shape[0], m))
    else:
        context = np.random.rand(x.shape[0], m)
    x = np.concatenate((x, context), axis=1)
    # reshape x matrix to make all samples matrixes (4, 1), not vector (4, )
    x = x.reshape(x.shape[0], 1, x.shape[1])
    w1 = (np.random.rand(p + m, m) * 2 - 1) / 10
    w2 = (np.random.rand(m, 1) * 2 - 1) / 10
    # this code learn for each sample
    for j in range(max_iter):
        error_all = 0
        if code_for_learning[1] == "1":
            x[:, :, -m:] = 0
        for i in range(x.shape[0]):
            hidden_layer = np.matmul(x[i], w1)
            output = np.matmul(hidden_layer, w2)
            dy = output - y[i]
            w1 -= alpha * dy * np.matmul(x[i].transpose(), w2.transpose())
            w2 -= alpha * dy * hidden_layer.transpose()
            try:
                x[i + 1][-m:] = hidden_layer
            except:
                pass
            # print("x=", x[i], "etalon", y[i], "result=", output)
        for i in range(x.shape[0]):
            hidden_layer = np.matmul(x[i], w1)
            output = np.matmul(hidden_layer, w2)
            dy = output - y[i]
            error_all += (dy ** 2)[0]
        print(j + 1, " ", error_all[0])
        if error_all <= error:
            break
    k = y[-1].reshape(1)
    X = x[-1, 0, :-m]
    out = []
    for i in range(5):
        X = X[1:]
        train = np.concatenate((X, k))
        X = np.concatenate((X, k))
        train = np.append(train, np.array([0] * m))
        if code_for_training[0]:
            train[-m:] = 0
        hidden_layer = np.matmul(train, w1)
        output = np.matmul(hidden_layer, w2)
        k = output
        out.append(k[0])
    return out


if __name__ == "__main__":
    print(
        start(
            sequence=[1, 2, 5, 15, 52, 203, 877, 4140, 21147],
            p=5,
            error=0.00000001,
            max_iter=1000000,
            code_for_learning="11",
            code_for_training="11",
        )
    )

