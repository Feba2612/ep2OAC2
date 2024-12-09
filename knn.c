#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void carregar_matriz(const char *filename, float **matriz, int *linhas, int *colunas) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Erro ao abrir o arquivo %s\n", filename);
        exit(1);
    }

    fscanf(file, "%d %d", linhas, colunas);
    *matriz = (float *)malloc((*linhas) * (*colunas) * sizeof(float));

    for (int i = 0; i < (*linhas); i++) {
        for (int j = 0; j < (*colunas); j++) {
            fscanf(file, "%f", (*matriz + i * (*colunas) + j));
        }
    }
    fclose(file);
}

void salvar_matriz(const char *filename, float *matriz, int linhas, int colunas) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Erro ao abrir o arquivo %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < linhas; i++) {
        for (int j = 0; j < colunas; j++) {
            fprintf(file, "%f ", *(matriz + i * colunas + j));
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

float calcular_distancia(float *a, float *b, int tamanho) {
    float distancia = 0.0;
    for (int i = 0; i < tamanho; i++) {
        distancia += pow(a[i] - b[i], 2);
    }
    return sqrt(distancia);
}

void knn(float *xtrain, float *ytrain, float *xtest, float *ytest, int linhas_train, int colunas, int linhas_test, int k) {
    #pragma omp parallel for
    for (int i = 0; i < linhas_test; i++) {
        float distancias[linhas_train];
        for (int j = 0; j < linhas_train; j++) {
            distancias[j] = calcular_distancia(xtest + i * colunas, xtrain + j * colunas, colunas);
        }

        // Selecionar os K vizinhos mais próximos
        float soma = 0;
        for (int m = 0; m < k; m++) {
            int indice_min = 0;
            for (int n = 1; n < linhas_train; n++) {
                if (distancias[n] < distancias[indice_min]) {
                    indice_min = n;
                }
            }
            soma += ytrain[indice_min];
            distancias[indice_min] = INFINITY; // Marcar como visitado
        }
        ytest[i] = soma / k; // Média aritmética dos K vizinhos
    }
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Uso: %s <xtrain> <ytrain> <xtest> <ytest>\n", argv[0]);
        return 1;
    }

    float *xtrain, *ytrain, *xtest, *ytest;
    int linhas_train, colunas_train, linhas_test, colunas_test;
    int k = 3; // Definir o valor de K aqui

    // Carregar os arquivos fornecidos como parâmetros
    carregar_matriz(argv[1], &xtrain, &linhas_train, &colunas_train);
    carregar_matriz(argv[2], &ytrain, &linhas_train, &colunas_train);
    carregar_matriz(argv[3], &xtest, &linhas_test, &colunas_test);

    // Alocar memória para ytest
    ytest = (float *)malloc(linhas_test * sizeof(float));

    // Verificar se as dimensões são compatíveis
    if (colunas_train != colunas_test) {
        printf("Erro: o número de colunas de xtrain e xtest deve ser o mesmo.\n");
        return 1;
    }

    // Medir o tempo de execução
    clock_t inicio = clock();
    knn(xtrain, ytrain, xtest, ytest, linhas_train, colunas_train, linhas_test, k);
    clock_t fim = clock();
    double tempo_execucao = (double)(fim - inicio) / CLOCKS_PER_SEC;
    printf("Tempo de execução: %f segundos\n", tempo_execucao);

    // Salvar o ytest gerado
    salvar_matriz(argv[4], ytest, linhas_test, 1);

    // Liberar memória
    free(xtrain);
    free(ytrain);
    free(xtest);
    free(ytest);

    return 0;
}