import pandas as pd
import h5py
import matplotlib.pyplot as plt
import yaml
import cantera as ct
import os

from matplotlib.backends.backend_pdf import PdfPages

def generate_plots(model_name, model_path, results_path, spec_keys_file, data_path, plot_path):

    sol = ct.Solution(model_path + model_name)

    h5_list = os.listdir(results_path)
    experimental_files = os.listdir(data_path)

    spec_names_model = (sol.species_names)

    for file in experimental_files:
        if os.path.exists(data_path+file):
            print(f"Loading {file}")
            with open(data_path+file, 'r') as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
            
        else:
            print(f"Couldn't find {data_path+file}")
            
        if os.path.exists(spec_keys_file):
            with open(spec_keys_file,'r') as k:
                key = yaml.load(k, Loader=yaml.SafeLoader)
        else:
            print(f"Couldn't find {spec_keys_file}")
            
        species = data['datapoints'][0]['outlet-composition']['species']
        csvfile = data['datapoints'][0]['csvfile']

        with PdfPages(plot_path+'jsr_plots.pdf') as plot_pdf:
            for sp in species:
                try:
                    name_in_model = key[model_name][sp['species-name']]
                    
                    temps = []
                    concs = []
                
                    ## get the position in which this species is listed in the model
                    i = 0
                    for name in spec_names_model:
                        if(name == name_in_model):
                            break
                        else:
                            i = i+1
                    ## iterate through all results files (each for a single temp)
                    for h5 in h5_list:
                        f = h5py.File(results_path+h5,'r')
                        dset = f['simulation']

                        temp = dset[1][1]
                        conc = (dset[-1][4][i])

                        concs.append(conc)
                        temps.append(temp)

                        f.close()
                        
                    temps, concs = zip(*sorted(zip(temps,concs))) ## sort both lists   
                        
                    plt.figure()
                    plt.plot(temps, concs, linestyle='solid')
                    plt.title(sp['species-name'] + ' concentration')

                    exp = pd.read_csv(csvfile)
                    temps = exp['Temperature']
                    concs = exp[sp['species-name']]

                    plt.scatter(temps,concs)
                    plt.legend(['simulated', 'experimental'])
                    plt.xlabel('Temperature (K)')
                    plt.ylabel('Mole Fraction')

                    plot_pdf.savefig()
                    plt.close()
                
                except KeyError:
                    continue