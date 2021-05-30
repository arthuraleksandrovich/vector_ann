"""Model testing and validation"""
import matplotlib.pyplot as plt
import numpy as np
import time

def validate_model(model_generator, datasets_generator, epochs=50, loss_name="mean_squared_error", measure_name="val_mean_squared_error", \
                   print_summary=True):
    """K-fold validation of model"""
    loss_history_sum = np.single(0)
    measure_history_sum = np.single(0)
    elapsed_time_sum = 0
    
    k = 0
    for train_dataset, validation_dataset in datasets_generator:
        model = model_generator()
    
        start_time = time.perf_counter()
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            verbose=0,
            use_multiprocessing = True
        )
        elapsed_time_sum += time.perf_counter() - start_time
        
        loss_history_sum += np.array(history.history[loss_name], dtype=np.float64)
        measure_history_sum += np.array(history.history[measure_name], dtype=np.float64)
        k += 1
    
    loss_history_avr = loss_history_sum / k
    measure_history_avr = measure_history_sum / k
    
    # Visualize error
    x = range(1, loss_history_avr.shape[0]+1)
    plt.plot(x, loss_history_avr, 'b')
    plt.plot(x, measure_history_avr, 'r--')
    plt.legend(['training loss', 'validation loss'])
    plt.show() 
    # Print statistics
    print(f'Elapsed k-fold validation time: {elapsed_time_sum:.5f} sec')
    if print_summary:
        model.summary()
    

def validate_model_multiple(model_generator, datasets_generator_fun, epochs=50, num_tries=5, \
                            loss_name="mean_squared_error", measure_name="val_mean_squared_error", \
                            print_data=False):
    """K-fold validation of model in multiple tries"""
    loss_history_sum = np.single(0)
    measure_history_sum = np.single(0)
    measure_history_worst = None
    measure_history_best = None
    last_measure_worst = None
    last_measure_best = None
    last_measures = []
    elapsed_time_sum = 0
    
    for t in range(num_tries):
        measure_history_sum_single = np.single(0)
        k = 0
        datasets_generator = datasets_generator_fun()
        for train_dataset, validation_dataset in datasets_generator:
            model = model_generator()
            
            start_time = time.perf_counter()
            history = model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=validation_dataset,
                verbose=0,
                use_multiprocessing = True
            )
            elapsed_time_sum += time.perf_counter() - start_time
            
            measure = np.array(history.history[measure_name], dtype=np.float64)
            best_measure = measure.min()
            last_measures.append(best_measure)
            
            loss_history_sum += np.array(history.history[loss_name], dtype=np.float64)
            measure_history_sum_single += measure
            
            k += 1
        
        measure_history_sum += measure_history_sum_single
        measure_history_avr_single = measure_history_sum_single / k
        best_measure = measure_history_avr_single.min()
        
        if last_measure_worst is None or best_measure > last_measure_worst:
            last_measure_worst = best_measure
            measure_history_worst = measure_history_avr_single
        if last_measure_best is None or best_measure < last_measure_best:
            last_measure_best = best_measure
            measure_history_best = measure_history_avr_single
    
    elapsed_time_avr = elapsed_time_sum / num_tries
    loss_history_avr = loss_history_sum / num_tries / k
    measure_history_avr = measure_history_sum / num_tries / k
    
    # Visualize error
    x = range(1, loss_history_avr.shape[0]+1)
    plt.subplot(2, 1, 1)
    plt.plot(x, loss_history_avr, 'b')
    plt.plot(x, measure_history_avr, 'y--')
    plt.plot(x, measure_history_worst, 'r--')
    plt.plot(x, measure_history_best, 'g--')
    plt.legend(['training loss avr.', 'validation loss avg.', 'validation loss worst', 'validation loss best'])
    plt.subplot(2, 1, 2)
    plt.boxplot(last_measures)
    # Print statistics
    print(f'Average elapsed k-fold validation time: {elapsed_time_avr:.5f} sec')
    if print_data:
        print('Last measures:', last_measures)
        print('Loss history average:', loss_history_avr)
        print('Measure history average:', measure_history_avr)
        print('Measure history worst:', measure_history_worst)
        print('Measure history best:', measure_history_best)
    

def test_model(model, train_dataset, test_dataset, epochs=50, loss_name="mean_squared_error", measure_name="val_mean_squared_error", \
               print_summary=True):
    """Testing of model"""
    start_time = time.perf_counter()
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        verbose=0,
        use_multiprocessing = True
    )
    elapsed_time = time.perf_counter() - start_time
    
    
    loss = history.history[loss_name]
    measure = history.history[measure_name]
    
    # Visualize error
    x = range(1, len(loss)+1)
    plt.plot(x, loss, 'b')
    plt.plot(x, measure, 'r--')
    plt.legend(['training loss', 'testing loss'])
    plt.show() 
    # Print statistics
    print(f'Elapsed training time: {elapsed_time:.5f} sec')
    if print_summary:
        model.summary()
    
# Testing of model in multiple runs
def test_model_multiple(model_generator, train_dataset, test_dataset, epochs=50, num_tries=10, \
                        loss_name="mean_squared_error", measure_name="val_mean_squared_error", \
                        print_data=False):
    """Testing of model in multiple tries"""
    loss_history_sum = np.single(0)
    measure_history_sum = np.single(0)
    measure_history_worst = None
    measure_history_best = None
    last_measure_worst = None
    last_measure_best = None
    last_measures = []
    elapsed_time_sum = 0
    
    for t in range(num_tries):
        model = model_generator()
        
        start_time = time.perf_counter()
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=test_dataset,
            verbose=0,
            use_multiprocessing = True
        )
        elapsed_time_sum += time.perf_counter() - start_time
        
        loss_history_sum += np.array(history.history[loss_name], dtype=np.float64)
        
        measure_history = np.array(history.history[measure_name], dtype=np.float64)
        best_measure = measure_history.min()
        if last_measure_worst is None or best_measure > last_measure_worst:
            last_measure_worst = best_measure
            measure_history_worst = measure_history
        if last_measure_best is None or best_measure < last_measure_best:
            last_measure_best = best_measure
            measure_history_best = measure_history
        measure_history_sum += measure_history
        
        last_measures.append(best_measure)
    
    elapsed_time_avr = elapsed_time_sum / num_tries
    loss_history_avr = loss_history_sum / num_tries
    measure_history_avr = measure_history_sum / num_tries
    
    # Visualize error
    x = range(1, loss_history_avr.shape[0]+1)
    plt.subplot(2, 1, 1)
    plt.plot(x, loss_history_avr, 'b')
    plt.plot(x, measure_history_avr, 'y--')
    plt.plot(x, measure_history_worst, 'r--')
    plt.plot(x, measure_history_best, 'g--')
    plt.legend(['training loss avr.', 'testing loss avg.', 'testing loss worst', 'testing loss best'])
    plt.subplot(2, 1, 2)
    plt.boxplot(last_measures)
    # Print statistics
    print(f'Average elapsed training time: {elapsed_time_avr:.5f} sec')
    if print_data:
        print('Last measures:', last_measures)
        print('Loss history average:', loss_history_avr)
        print('Measure history average:', measure_history_avr)
        print('Measure history worst:', measure_history_worst)
        print('Measure history best:', measure_history_best)