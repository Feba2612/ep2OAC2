#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// calcula a distância euclidiana entre dois vetores de tamanho w. Ela percorre os elementos de p1 e p2,
// soma as diferenças quadradas e retorna a raiz quadrada da soma.
double calcular_distancia(float *p1, float *p2, int w) {
    double soma = 0.0;
    for (int i = 0; i < w; i++) {
        soma += pow(p1[i] - p2[i], 2);
    }
    return sqrt(soma);
}

//normaliza cada coluna da matriz X. A normalização coloca os valores entre 0 e 1,
// garantindo que cada atributo tenha a mesma escala. A normalização é paralelizada com OpenMP.
void normalizar_matriz(float **X, int linhas, int colunas) {
    for (int j = 0; j < colunas; j++) {
        float min = X[0][j], max = X[0][j];
        for (int i = 1; i < linhas; i++) {
            if (X[i][j] < min) min = X[i][j];
            if (X[i][j] > max) max = X[i][j];
        }
        if (max - min > 0) {
            for (int i = 0; i < linhas; i++) {
                X[i][j] = (X[i][j] - min) / (max - min);
            }
        }
    }
}

// implementa o KNN. Para cada instância de xtest, calcula-se a distância para cada exemplo em X_train. 
//Depois, as k menores distâncias são ordenadas, e a função retorna a média dos valores correspondentes em ytrain.
float knn(float **X_train, float *ytrain, float *xtest, int ntrain, int w, int k) {
    double *distancias = (double *)malloc(ntrain * sizeof(double));
    int *indices = (int *)malloc(ntrain * sizeof(int));

    // Calcular distâncias entre xtest e todas as instâncias de X_train
    for (int i = 0; i < ntrain; i++) {
        distancias[i] = calcular_distancia(X_train[i], xtest, w);
        indices[i] = i;
    }

    // Ordenar distâncias para encontrar os K menores
    for (int i = 0; i < k; i++) {
        for (int j = i + 1; j < ntrain; j++) {
            if (distancias[j] < distancias[i]) {
                double temp_dist = distancias[i];
                distancias[i] = distancias[j];
                distancias[j] = temp_dist;

                int temp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = temp_idx;
            }
        }
    }

    // Calcular a média dos valores de ytrain correspondentes aos K menores
    float soma_y = 0.0;
    for (int i = 0; i < k; i++) {
        int idx = indices[i];
        soma_y += ytrain[idx];
    }

    free(distancias);
    free(indices);

    return soma_y / k;
}


// Conta o número de linhas em um arquivo. Usada para determinar 
//quantas amostras de treino e teste existem.
int contar_linhas(char *arquivo) {
    FILE *fp = fopen(arquivo, "r");
    if (fp == NULL) {
        printf("Erro ao abrir o arquivo %s\n", arquivo);
        exit(1);
    }
    int linhas = 0;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), fp)) linhas++;
    fclose(fp);
    return linhas;
}

// Função para ler dados do arquivo e armazenar no vetor x
void ler_dados(char *arquivo, float *x, int n_linhas) {
    FILE *fp = fopen(arquivo, "r");
    if (fp == NULL) {
        printf("Erro ao abrir o arquivo %s\n", arquivo);
        exit(1);
    }
    for (int i = 0; i < n_linhas; i++) fscanf(fp, "%f", &x[i]);
    fclose(fp);
}

// Função para gerar as janelas deslizantes de `X_train` e calcula os vetores de `ytrain`.
// A função percorre o vetor de treino (xtrain), criando subconjuntos de tamanho w, 
// e calcula a média dos próximos h valores para o vetor ytrain.
void gerar_X_y_train(float *xtrain, float **X_train, float *ytrain, int ntrain, int w, int h) {
    for (int i = 0; i < ntrain - w - h; i++) {
        for (int j = 0; j < w; j++) {
            X_train[i][j] = xtrain[i + j];
        }
        float soma = 0.0;
        for (int j = 0; j < w; j++) {
            soma += xtrain[i + j + h];
        }
        ytrain[i] = soma / w;
    }
}

// Gera janelas deslizantes para X_test.
void gerar_X_test(float *xtest, float **X_test, int ntest, int w) {
    for (int i = 0; i < ntest - w; i++) {
        for (int j = 0; j < w; j++) {
            X_test[i][j] = xtest[i + j];
        }
    }
}

// Função para salvar resultados.
void salvar_dados(char *arquivo, float *y, int n_linhas) {
    FILE *fp = fopen(arquivo, "w");
    if (fp == NULL) {
        printf("Erro ao salvar o arquivo %s\n", arquivo);
        exit(1);
    }
    for (int i = 0; i < n_linhas; i++) fprintf(fp, "%.2f\n", y[i]);
    fclose(fp);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Uso: %s <arquivo_train> <arquivo_test>\n", argv[0]);
        return 1;
    }

    char *arquivo_train = argv[1];
    char *arquivo_test = argv[2];

    int k = 5;
    int w = 3; // Tamanho da janela de previsão
    int h = 1; // Passo de previsão

    int ntrain = contar_linhas(arquivo_train);
    int ntest = contar_linhas(arquivo_test);

    float *xtrain = (float *)malloc(ntrain * sizeof(float));
    float *xtest = (float *)malloc(ntest * sizeof(float));

    float **X_train = (float **)malloc((ntrain - w - h) * sizeof(float *));
    for (int i = 0; i < ntrain - w - h; i++) {
        X_train[i] = (float *)malloc(w * sizeof(float));
    }
    float *ytrain = (float *)malloc((ntrain - w - h) * sizeof(float));

    float **X_test = (float **)malloc((ntest - w) * sizeof(float *));
    for (int i = 0; i < ntest - w; i++) {
        X_test[i] = (float *)malloc(w * sizeof(float));
    }

    ler_dados(arquivo_train, xtrain, ntrain);
    ler_dados(arquivo_test, xtest, ntest);

    gerar_X_y_train(xtrain, X_train, ytrain, ntrain, w, h);
    gerar_X_test(xtest, X_test, ntest, w);

    normalizar_matriz(X_train, ntrain - w - h, w);
    normalizar_matriz(X_test, ntest - w, w);

    clock_t start = clock();

    float *ytest = (float *)malloc((ntest - w) * sizeof(float));
    for (int i = 0; i < ntest - w; i++) {
        ytest[i] = knn(X_train, ytrain, X_test[i], ntrain - w - h, w, k);
    }

    clock_t end = clock();
    double tempo_execucao = ((double)(end - start)) / CLOCKS_PER_SEC;

    salvar_dados("ytrain.txt", ytrain, ntrain - w - h);
    salvar_dados("ytest.txt", ytest, ntest - w);

    printf("Tempo de execução (apenas previsão): %.6f segundos\n", tempo_execucao);

    for (int i = 0; i < ntrain - w - h; i++) free(X_train[i]);
    for (int i = 0; i < ntest - w; i++) free(X_test[i]);
    free(xtrain);
    free(xtest);
    free(X_train);
    free(X_test);
    free(ytrain);
    free(ytest);

    return 0;
}
