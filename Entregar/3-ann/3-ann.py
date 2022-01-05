"""
Practica 2: Exercises of ANN classifiers
Autor: Pablo García Fernández.  

"""

def pr3(dataset_name, dataset_train=None, dataset_test=None):
    """ Función que ejecuta los 3 ejercicios mencionados
    en la practica 3.
    - ej2: Calcular acc., kappa y cm usando MLP y ELM. Todo el dataset.
           como train/test. Configuración de hiperparámetros por defecto.
           Ver que pasa al cambiar nº de neuronas ocultas.
    - ej3: Repetir con cross-validation k=4. Sintonizar params.
    - ej4: Usar MLP y ELM en LBP y CooCu datasets

    dataset_name se emplea para hepatitis y wine datasets (donde no hay 
    separacion entre train y test). dataset_train y dataset_test se emplea
    para LBP y Coocur (donde train y test ya están separados)
    """

    print("=============================================")
    print(f"   Ejecucion pr2 sobre {dataset_name} dataset!")
    print("=============================================\n")

    # EJERCICIO 5 -----------------------------------------------------------------
    # LDA básico sobre LBP y CooCU
    if dataset_name == 'LBP' or dataset_name == 'Coocur':