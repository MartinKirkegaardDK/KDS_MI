from utils.plot_outlier_neurons import plot_neuron_contributions, common_outliers
import pickle

def main(num_layers:int, hook_points:list, model_name_temp:str):
    
    
    with open(f'results/data/neuron_contributions/{model_name_temp}.pkl', 'rb') as f:
        neuron_contributions = pickle.load(f)

    with open(f'results/data/common_indices/{model_name_temp}.pkl', 'rb') as f:
        obj = pickle.load(f)
     
 
    print(neuron_contributions.keys())

    #common_outliers(model_name_temp,obj)


    plot_neuron_contributions(num_layers, hook_points, model_name_temp, neuron_contributions)


