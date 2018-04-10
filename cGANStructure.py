def Structure(name):

    input_dim = 20
    G_dense = 160
    D_dense = 80

    if name == "Breast":
        input_dim = 70
        G_dense = 90
        D_dense = 45

    elif name == "ecoli":
        input_dim = 35
        G_dense = 40
        D_dense = 20

    elif name == "glass":
        input_dim = 25
        G_dense = 70
        D_dense = 35

    elif name == "haberman":
        input_dim = 10
        G_dense = 20
        D_dense = 10

    elif name == "heart":
        input_dim = 65
        G_dense = 80
        D_dense = 40

    elif name == "iris":
        input_dim = 20
        G_dense = 25
        D_dense = 15

    elif name == "libras":
        input_dim = 180
        G_dense = 540
        D_dense = 270

    elif name == "liver":
        input_dim = 50
        G_dense = 35
        D_dense = 20

    elif name == "segment":
        input_dim = 80
        G_dense = 160
        D_dense = 80

    elif name == "Vehicle":
        input_dim = 70
        G_dense = 100
        D_dense = 55

    elif name == "wine":
        input_dim = 130
        G_dense = 65
        D_dense = 40

    return input_dim, G_dense, D_dense
