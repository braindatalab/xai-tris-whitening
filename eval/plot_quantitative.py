import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
plt.style.use('seaborn-colorblind')
import sys
import os
# import pandas as pd
import random
import pickle as pkl

from math import floor
import torch
import seaborn as sns
import pandas as pd
from collections import Counter
from utils import load_json_file
from pathlib import Path

from glob import glob

plt.style.use('seaborn-poster')  # This makes the plots larger and more legible
sns.set_context("talk")  # Also increases the text size for seaborn plots

NAME_MAPPING = {
    "PFI": "PFI",
    "Integrated Gradients": "Int. Grad.",
    "Saliency": "Saliency",
    "DeepLift": "DeepLift",
    "DeepSHAP": "DeepSHAP",
    "Gradient SHAP": "Grad\nSHAP",
    "Guided Backprop": "Guided\nBackprop.",
    "Guided GradCAM": "Guided\nGradCAM",
    "Deconvolution": "Deconv",
    "Shapley Value Sampling": "Shapley\nSampling",
    "LIME": "LIME",
    "Kernel SHAP": "Kernel\nSHAP",
    "LRP": "LRP",
    "pattern.net": "PatternNet",
    "pattern.attribution": "PatternAttr.",
    "deep_taylor": "DTD",
    "sobel": "Sobel",
    "laplace": "Laplace"
}

abbrev_dict = {
    'linear\nuncorrelated': 'LIN WHITE',
    'linear\ncorrelated': 'LIN CORR',
    'multiplicative\nuncorrelated': 'MULT WHITE',
    'multiplicative\ncorrelated': 'MULT CORR',
    'translations\nrotations\nuncorrelated': 'RIGID WHITE',
    'translations\nrotations\ncorrelated': 'RIGID CORR',
    'xor\nuncorrelated':'XOR WHITE',
    'xor\ncorrelated': 'XOR CORR',
}

def extract_whitening_method(filename):
    whitening_methods = [
        'sphering', 
        'cholesky_whitening', 
        'optimal_signal_preserving_whitening', 
        'partial_regression', 
        'symmetric_orthogonalization'
    ]
    # Split the filename by the known suffix to isolate the whitening method part
    base_part = filename.split('_quantitative_results')[0]
    # print("base_part:", base_part)
    for method in whitening_methods:
        if method in base_part:
            return method
    return 'no_whitening'
    
    
def enhance_plots(g, metric, full_range=(0, 1)):
    # Setting the y-axis limits for all plots
    for ax in g.axes.flat:
        ax.set_ylim(full_range)
        ax.set_ylabel(metric)
        ax.grid(True)

    # Reduce space between title and first row of plots
    plt.subplots_adjust(top=0.85, hspace=0.3)  # Adjust 'top' and 'hspace' as needed


def quantitative_data_plot(boxplot_dict, metric='Precision', predictions='correct', out_dir=None, exp_size=25, palette='colorblind'):
    background_mapping = {
        'uncorrelated - no_whitening': 'uncorrelated',
        'correlated - no_whitening': 'correlated',
        'correlated - cholesky_whitening': 'cholesky',
        'correlated - partial_regression': 'partial regression',
        'correlated - sphering': 'sphering',
        'correlated - symmetric_orthogonalization': 'symmetric orthogonalization',
        'correlated - optimal_signal_preserving_whitening': 'optimal signal preserving'
    }

    row_order = ['linear', 'multiplicative', 'xor', 'translations_rotations']
    col_order = ['LLR', 'MLP', 'CNN']
   
    data = pd.DataFrame(boxplot_dict)
    data = data[~(data['background'].str.contains('uncorrelated') & ~data['background'].str.contains('no_whitening'))]  
    # Get summary statistics for numerical columns
    data_description = data.describe()

    # Get information about data types and missing values
    data_info = data.info()

    # Display unique values for non-numeric data
    unique_values = {col: data[col].unique() for col in data.select_dtypes(include=['object', 'category']).columns}
    
    print("numerical data description:", data_description)
    print("-----------------------------------")
    print("data types and missing values info:", data_info)
    print("-----------------------------------")
    print("unique values for non-numeric data:", unique_values)
    print("-----------------------------------")
    
    data['background'] = data['background'].map(background_mapping)
    
    sns.boxplot(x='background', y='EMD', data=data)
    plt.savefig('test.png', format='png', bbox_inches='tight', dpi=300)
     
    background_order = ['uncorrelated', 'correlated', 'sphering', 'symmetric orthogonalization', 'cholesky', 'partial regression', 'optimal signal preserving']
    grouped_data = data.groupby('background').agg({'Precision': 'mean', 'EMD': 'mean'}).reset_index()
    print("grouped_data", grouped_data.head())
    # Categorize the 'background' column to enforce the order
    grouped_data['background'] = pd.Categorical(grouped_data['background'], categories=background_order, ordered=True)
    print("grouped_data after pd.Categorical", grouped_data.head())
    # Sort the DataFrame based on the 'background' column
    grouped_data = grouped_data.sort_values('background')
    print("grouped_data after sorting", grouped_data.head())
    # Setting up the plotting area
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))

    # For Precision
    axes[0].boxplot(
        x=[data.loc[data['background'] == bg, 'Precision'] for bg in background_order],
        vert=False,  # Horizontal box plots
        labels=background_order,
        showfliers=False,
        patch_artist=True,  # Fill with color
        medianprops={'color': 'black'},  # Median line properties
        boxprops=dict(facecolor='skyblue', color='skyblue'),  # Box properties
    )
    axes[0].set_title('Average Precision by Background Type', fontsize=14)
    axes[0].set_xlabel('Average Precision', fontsize=12)
    axes[0].set_ylabel('Background Type', fontsize=12)
    axes[0].invert_yaxis()
    axes[0].grid(True)
    # For EMD
    axes[1].boxplot(
        x=[data.loc[data['background'] == bg, 'EMD'] for bg in background_order],
        vert=False,  # Horizontal box plots
        labels=background_order,
        showfliers=False,
        patch_artist=True,  # Fill with color
        medianprops={'color': 'black'},  # Median line properties
        boxprops=dict(facecolor='salmon', color='salmon'),  # Box properties
    )
    axes[1].grid(True)
    axes[1].set_title('Average EMD by Background Type', fontsize=14)
    axes[1].set_xlabel('Average EMD', fontsize=12)
    axes[1].invert_yaxis()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save and display the plots
    plt.savefig('whitening_effects_plots.png', format='png', bbox_inches='tight', dpi=300)

    # Plot settings
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(20, 15))

    for xai_method in data['Method'].unique():
        print("xai_method:", xai_method)
        method_df = data[data['Method'] == xai_method]
        if (metric == 'Precision'):
            g = sns.catplot(
                x="background",
                y=metric,
                hue="background",
                palette='colorblind',
                col='model',
                row='scenario',
                data=method_df,
                kind="point",
                order=list(background_mapping.values()),
                row_order=row_order,
                col_order=col_order,
                legend_out=True,
                height=5,  
                aspect=1.3,
                # sharey='row',
                errorbar='sd',
                hue_order=list(background_mapping.values())
            )
        else:
            g = sns.catplot(
                x="background",
                y=metric,
                hue="background",
                palette='colorblind',
                col='model',
                row='scenario',
                data=method_df,
                kind="box",
                notch=True,
                order=list(background_mapping.values()),
                row_order=row_order,
                col_order=col_order,
                showfliers=False,
                legend_out=True,
                height=5,  
                aspect=1.3,
                sharey='row',
                ci='sd',
                hue_order=list(background_mapping.values())
            )
        g.set_xticklabels(rotation=45, horizontalalignment='right')
        g.set_titles("{col_name} - {row_name}", size=16)
        
        ax = g.axes
        for row_idx in range(ax.shape[0]):
            for col_idx in range(ax.shape[1]):
            # Hide specific subplots based on their indices
                if row_idx == 0 and col_idx == 0:
                    ax[row_idx][col_idx].set_xlabel('background', fontsize=14)
                if row_idx > 0 and col_idx == 0:
                    ax[row_idx, col_idx].set_visible(False)
            
        # Iterate over each subplot to adjust font size and grid
        min_val = data[metric].min() - 0.05 * data[metric].min()
        max_val = data[metric].max() + 0.05 * data[metric].max()
        g.axes[0, 0].set_ylabel('')
        for ax in g.axes.flat:
            ax.set_xlabel('')
            ax.grid(True)  # Add gridlines for better readability
            ax.tick_params(axis='y', which='both', labelleft=True)
            ax.set_ylim(min_val, max_val)
        # Adjust spacing between plots
        plt.subplots_adjust(hspace=0.3, wspace=0.4)
        g.figure.suptitle(f'{metric} distribution across backgrounds for {xai_method}', fontsize=20, y=1.03)
        
        
        
        handles, labels = g.axes.flat[0].get_legend_handles_labels()
        g.figure.legend(handles, labels, title='Whitening Effect', loc='lower left', bbox_to_anchor=(0.1, 0.4), fontsize='large', title_fontsize='x-large', frameon=True, shadow=True, fancybox=True)
        # Save the plot
        if out_dir:
            filename = f'{out_dir}/{metric}_{xai_method}.png'
            g.savefig(filename, format='png', bbox_inches='tight', dpi=300)
        plt.close()
       

def model_plot(boxplot_dict, metric='Precision', predictions='correct', out_dir=None, exp_size=25, palette='colorblind'):
    plt.rcParams['ytick.labelsize'] = 16
    ncol = 1
    bbox_anchor = (-1.65, 5)
    figsize = (30,25)
    
    x_order=[
        "PFI", "Int. Grad.", "Saliency", "DeepLift", "DeepSHAP", "Grad SHAP", 
        "Deconv", "Shapley Value Sampling",
        "LIME", "Kernel SHAP", "LRP", "PatternNet", "PatternAttr.",
        "DTD", "Guided Backprop.", "Guided GradCAM", "Sobel", "Laplace", "rand","x"
    ]
    
    row_order = [
        'linear\nuncorrelated', 'linear\ncorrelated', 'multiplicative\nuncorrelated', 'multiplicative\ncorrelated',
        'translations\nrotations\nuncorrelated','translations\nrotations\ncorrelated','xor\nuncorrelated','xor\ncorrelated'
    ]
    col_order = ['LLR', 'MLP', 'CNN']
    
    dd = pd.DataFrame(boxplot_dict)
     # Get summary statistics for numerical columns
    data_description = dd.describe()

    # Get information about data types and missing values
    data_info = dd.info()

    # Display unique values for non-numeric data
    unique_values = {col: dd[col].unique() for col in dd.select_dtypes(include=['object', 'category']).columns}
    
    print("numerical data description:", data_description)
    print("-----------------------------------")
    print("data types and missing values info:", data_info)
    print("-----------------------------------")
    print("unique values for non-numeric data:", unique_values)
    print("-----------------------------------")
    f = sns.catplot(x="Method", y=metric, hue="Methods", palette=palette, col='model', row='datasets', data=dd, legend_out=True, kind='box', row_order=row_order, col_order=col_order, order=x_order, hue_order=x_order)
    print(f)
    legend_handles = f.legend.legendHandles
    f.fig.figsize = figsize
    plt.close(fig=f.fig)
    
    if 'Precision' in metric or 'AUROC' in metric:
        g = sns.catplot(x="Methods", y=metric, palette=palette, row='datasets', col='model', data=dd, errorbar='sd', kind="point", row_order=row_order, col_order=col_order, order=x_order, hue_order=x_order)
    else:
        g = sns.catplot(x="Methods", y=metric, palette=palette, row='datasets', col='model', data=dd, ci='sd', kind="box", notch=True, row_order=row_order, col_order=col_order, order=x_order, showfliers=False, hue_order=x_order)
    # if model == 'LLR':
    #     g = sns.catplot(x="Method", y=metric, palette='colorblind', col='model', data=dd, ci='sd', kind="point")
    # else:
    # g = sns.catplot(x="Methods", y=metric, palette=palette, col='model', row='datasets', data=dd, ci='sd', kind="point")
    
    # g = sns.catplot(x="Methods", y=metric, palette=palette, row='datasets', col='model', data=dd, ci='sd', kind="box", notch=True, row_order=row_order, col_order=col_order, order=x_order, showfliers=False, hue_order=x_order)
    g.fig.set_figwidth(figsize[0])
    g.fig.set_figheight(figsize[1])
    
    ax = g.axes
    configure_axes(ax, metric=metric)
    
    plt.legend(handles=legend_handles, bbox_to_anchor=bbox_anchor, prop={'size': 20},
               ncol=ncol, fancybox=True, loc='upper center', borderaxespad=0., facecolor='white', framealpha=0.5)

    plt.subplots_adjust(top=0.92)
    plt.subplots_adjust(wspace=0.05, hspace=0.12)    
    plt.suptitle('Symmetric Orthogonalization: ' + metric , fontsize=18, x=0.5, y=0.95, fontweight='bold')
    if out_dir:
        plt.savefig(f'{out_dir}/model_full_{metric}_{predictions}_{exp_size}.png', format='png', bbox_inches='tight', dpi=300)


def configure_axes_modified(ax, metric='Precision'):
    for row_idx in range(ax.shape[0]):
        for col_idx in range(ax.shape[1]):
            # Hide specific subplots based on their indices
            if row_idx > 0 and col_idx == 0:
                ax[row_idx, col_idx].set_visible(False)
            else:
                # Set the x and y labels for the subplot that remains visible
                # This will be the subplot in the first row and first column after applying your condition
                ax[row_idx, col_idx].set_xlabel(metric, fontdict={'fontsize': 16})
                ax[row_idx, col_idx].set_ylabel('Value', fontdict={'fontsize': 20})  # Change 'Value' to your actual y-axis label

                # You can also adjust the position of the y-axis label if needed
                ax[row_idx, col_idx].yaxis.set_label_coords(-0.1, 0.5)  # Adjust the coordinates as necessary

            # Apply other universal settings here, e.g., removing spines, setting grid visibility, etc.
            sns.despine(ax=ax[row_idx, col_idx], top=True, right=True, left=False, bottom=False)
            ax[row_idx, col_idx].grid(True)


def configure_axes(ax, metric='EMD'):
    print(ax.shape)
    for row_idx in range(ax.shape[0]):
        t = ax[row_idx, 0].get_title(loc='center')
        name_of_metric = t.split('|')[0].split('=')[-1].strip()

        ax[row_idx, 0].set_ylabel(ylabel=abbrev_dict[name_of_metric].replace(' ', '\n'),
                                    fontdict={'fontsize': 20})
        if row_idx > 1:
            ax[row_idx, 0].yaxis.set_label_coords(0.9575,0.5)
        
        for col_idx in range(ax.shape[1]):
            if 0 == row_idx:
                t = ax[row_idx, col_idx].get_title(loc='center')
                new_title = t.split('|')[-1].strip().split('=')[-1].strip()
                ax[row_idx, col_idx].set_title(
                    label=new_title, fontdict={'fontsize': 22})
            else:
                ax[row_idx, col_idx].set_title(label='')
            ax[row_idx, col_idx].set_xlabel(xlabel='',
                                            fontdict={'fontsize': 16})
            labels = ax[row_idx, col_idx].get_xticklabels()
            ax[row_idx, col_idx].set_xticklabels('')
            ax[row_idx, col_idx].patch.set_edgecolor('black')
            
            sns.despine(ax=ax[row_idx, col_idx],
                        top=False, bottom=False, left=False, right=False)
            ax[row_idx, col_idx].grid(True)
            if metric=='EMD':
                ax[row_idx, col_idx].set_ylim(0.6, 1.0)
            if row_idx > 1:
                if col_idx == 0:
                    ax[row_idx, col_idx].xaxis.set_visible(False)
                    ax[row_idx, col_idx].yaxis.grid(False, which='both')
                    plt.setp(ax[row_idx, col_idx].spines.values(), visible=False)
                    ax[row_idx, col_idx].tick_params(left=False, labelleft=False)
                    ax[row_idx, col_idx].patch.set_visible(False)
                else:
                    if col_idx==1:
                        ax[row_idx, col_idx].tick_params(left=True, labelleft=True)
                        if metric=='EMD':
                            ax[row_idx, col_idx].set_yticks([0.62,0.7,0.8,0.9,0.98])
                            ax[row_idx, col_idx].set_yticklabels([0.6,0.7,0.8,0.9,1.0])
                        else:
                            ax[row_idx, col_idx].set_yticks([0.02,0.25,0.5,0.75,0.98])
                            ax[row_idx, col_idx].set_yticklabels([0.0,0.25,0.5,0.75,1.0])
                        

def main():
    config = load_json_file(file_path='eval/eval_config.json')
    out_dir = f'{config["out_dir"]}/figures'
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    random.seed(2022)
    palette = sns.color_palette('tab20')
    random.shuffle(palette)
    random.shuffle(palette)

    for exp_size in [config['num_experiments']]:
        llr_dict = {
            'Method': list(), 'Methods': list(), 'background': list(), 'scenario': list(), 'dataset': list(), 'datasets': list(),
            'Precision': list(), 'EMD': list(), 'Precision_4': list(), 'EMD_4': list()
        }
        mlp_dict = {
            'Method': list(), 'Methods': list(), 'background': list(), 'scenario': list(), 'dataset': list(), 'datasets': list(),
            'Precision': list(), 'EMD': list(), 'Precision_4': list(), 'EMD_4': list()
        }

        cnn_dict =  {
            'Method': list(), 'Methods': list(), 'background': list(), 'scenario': list(), 'dataset': list(), 'datasets': list(),
            'Precision': list(), 'EMD': list(), 'Precision_4': list(), 'EMD_4': list()
        }
        all_dicts = {'LLR': llr_dict, 'MLP': mlp_dict, 'CNN': cnn_dict}

        combined_dict = {
            'Method': list(), 'Methods': list(), 'background': list(), 'scenario': list(), 'dataset': list(), 'datasets': list(),
            'Precision': list(), 'EMD': list(),  'model': list(), 'models': list(),
        }

        new_dict = {
            'Method': list(), 'Methods': list(), 'background': list(), 'scenario': list(), 'dataset': list(), 'datasets': list(),
            'Precision': list(), 'EMD': list(), 'Precision_4': list(), 'EMD_4': list(), 'model': list(), 'models': list()
        }

        combined_dict_keras = {
            'Method': list(), 'Methods': list(), 'background': list(), 'scenario': list(), 'dataset': list(), 'datasets': list(),
            'Precision': list(), 'EMD': list(), 'model': list(), 'models': list()
        }

        used_keys_comb = combined_dict.keys()
        for i in range(exp_size):
            boxplot_dict = {'results': {}}
            boxplot_dict_keras = {'results': {}}
            
            # for whiteningMethod in ['noWhitening', 'cholesky_whitening', 'optimal_signal_preserving_whitening', 'partial_regression', 'sphering', 'symmetric_orthogonalization']:
            # for whiteningMethod in ['symmetric_orthogonalization']:
            for whiteningMethod in ['noWhitening', 'final']:
                torch_paths = glob(f'{config["out_dir"]}/final/*{whiteningMethod}_quantitative_results.pkl')
                # torch_paths = glob(f'{config["out_dir"]}/{whiteningMethod}/*_quantitative_results.pkl')
                # torch_paths = glob(f'{config["out_dir"]}/{whiteningMethod}/*_quantitative_results.pkl')
                for torch_path in torch_paths:
                    with open(torch_path, 'rb') as file:
                        torch_results = pkl.load(file)
                    for key, val in torch_results['results'].items():
                        boxplot_dict['results'][key] = val
                # keras_paths = glob(f'{config["out_dir"]}/{whiteningMethod}/*_quantitative_results_keras.pkl')
                keras_paths = glob(f'{config["out_dir"]}/final/*{whiteningMethod}_quantitative_results_keras.pkl')
                # keras_paths = glob(f'{config["out_dir"]}/{whiteningMethod}/*_quantitative_results_keras.pkl')
                for keras_path in keras_paths:
                    with open(keras_path, 'rb') as file:
                        keras_results = pkl.load(file)
                    for key, val in keras_results['results'].items():
                        boxplot_dict_keras['results'][key] = val

            for scenario, scenario_results in boxplot_dict['results'].items():
                whitening_method = extract_whitening_method(scenario)
                for model_name, model_results in scenario_results['correct'].items():
                    results_len = 0
                    keras_len = 0
                    for j, result in enumerate(model_results[:5]):
                        for key, value in result.items():
                            if key in used_keys_comb:
                                val = value
                                if model_name == 'CNN':
                                    if 'xor' not in scenario:
                                        keras_val = boxplot_dict_keras['results'][scenario]['correct'][model_name][j][key]    
                                else:
                                    keras_val = boxplot_dict_keras['results'][scenario]['correct'][model_name][j][key]
                                if key == 'Method' or key == 'Methods':
                                    val = [m.replace('\n', ' ') for m in value]
                                    if 'xor' in scenario and model_name == 'CNN':
                                        keras_val = []
                                    else:
                                        keras_val = [m.replace('\n', ' ') for m in boxplot_dict_keras['results'][scenario]['correct'][model_name][j][key]]
                                
                                if key == 'background':
                                    # Update the background value to include whitening method information
                                    updated_background_torch = [f"{bg} - {whitening_method}" for bg in val]
                                    updated_background_keras = [f"{bg} - {whitening_method}" for bg in keras_val]
                                    combined_dict['background'] += updated_background_torch
                                    combined_dict_keras['background'] += updated_background_keras
                                else:     
                                    combined_dict[key] += val
                                    combined_dict_keras[key] += keras_val
                        if model_name == 'CNN':
                            if 'xor' in scenario:
                                keras_len += 0
                            else:
                                keras_len += len(boxplot_dict_keras['results'][scenario]['correct'][model_name][j]['Methods'])        
                        else:
                            keras_len += len(boxplot_dict_keras['results'][scenario]['correct'][model_name][j]['Methods'])
                        results_len += len(result['Methods']) 
                        combined_dict['dataset'] += [m.replace('_','\n')+'\n'+n for m,n in zip(result["scenario"],result["background"])]
                        combined_dict['datasets'] += [m.replace('_','\n')+'\n'+n for m,n in zip(result["scenario"],result["background"])]

                        if 'xor' in scenario and model_name == 'CNN':
                            combined_dict_keras['dataset'] += []
                            combined_dict_keras['datasets'] += []
                        else:
                            combined_dict_keras['dataset'] += [m.replace('_','\n')+'\n'+n for m,n in zip(boxplot_dict_keras['results'][scenario]['correct'][model_name][j]["scenario"],boxplot_dict_keras['results'][scenario]['correct'][model_name][j]["background"])]
                            combined_dict_keras['datasets'] += [m.replace('_','\n')+'\n'+n for m,n in zip(boxplot_dict_keras['results'][scenario]['correct'][model_name][j]["scenario"],boxplot_dict_keras['results'][scenario]['correct'][model_name][j]["background"])]

                        new_dict['dataset'] += [m for m in result["scenario"]]
                        new_dict['datasets'] += [m for m in result["scenario"]]
                                
                    combined_dict['model'] += [model_name]*results_len
                    combined_dict['models'] += [model_name]*results_len

                    combined_dict_keras['model'] += [model_name]*keras_len
                    combined_dict_keras['models'] += [model_name]*keras_len
                        
        for key in combined_dict.keys():
            combined_dict[key] += combined_dict_keras[key]
        # baselines only calculated for one model, need to copy over to other model results to show up in box plots
        for method in ['Sobel', 'Laplace', 'x', 'rand']:
            method_inds_llr = np.where((np.array(combined_dict['Method']) == method) & (np.array(combined_dict['scenario']) == 'linear'))[0]
            method_inds_mlp = np.where(np.array(combined_dict['Method']) == method)[0]

            for model, method_inds in {'LLR': method_inds_llr, 'MLP': method_inds_mlp}.items():
                for key in combined_dict.keys():
                    if 'model' in key:
                        continue
                    duplicate_values = np.array(combined_dict[key])[method_inds].copy()

                    combined_dict[key] += list(duplicate_values)
                combined_dict['model'] += [model]*len(duplicate_values)
                combined_dict['models'] += [model]*len(duplicate_values)        
        for metric in ['EMD', 'Precision']:
            quantitative_data_plot(combined_dict, metric=metric, predictions='correct', out_dir=out_dir, exp_size=exp_size, palette=palette)
            # model_plot(combined_dict, metric=metric, predictions='correct', out_dir=out_dir, exp_size=exp_size, palette=palette)

if __name__ == '__main__':
    main()