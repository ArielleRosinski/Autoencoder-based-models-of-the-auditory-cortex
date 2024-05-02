def plot_model_comp(combined_data,title,start_idh, plot_labels, colours, pred_corr_lbl=False,mdn=True, idx_mdn=-1,ylim=True, nplots=4, pd=False, fontsize=13, start_idv=0):
    """Combined_data e.g., [pred_corr_ridge_untrained, pred_corr_ridge_PC, pred_corr_ridge_time_scrambled_ae, pred_corr_ridge_ae]
       plot_labels e.g., ['Untrained','PCA','Time-scrambled\n AE', 'AE']
       colours e.g., ['gray','royalblue', 'darkorange','mediumseagreen'] """


    boxprops = {'facecolor': 'none', 'edgecolor': 'black'}


    for i in range(nplots):
        axes = fig.add_subplot(gs[start_idv, i+start_idh])

        sns.stripplot(data=combined_data[i], orient='v', color=colours[i], alpha=0.5,  ax=axes,zorder=0)

        sns.boxplot(data=combined_data[i], ax=axes, orient='v', boxprops=boxprops, zorder=1e5,  width=0.1)

        axes.set_xlabel(plot_labels[i], fontsize=fontsize, rotation=45)
        axes.set_xticklabels([])
        if mdn and pd==False:
            axes.axhline(y=np.nanmedian(combined_data[idx_mdn]), color="black", linestyle="--")
        if mdn and pd:
            axes.axhline(y=np.nanmedian(pred_corr_poisson_nn_ae), color="black", linestyle="--")     

        
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        if ylim:
            axes.set_ylim(-0.45, 1.1)
        else:
            axes.set_ylim(-0.2, 1.1)
        if  i == 0:
            axes.tick_params(axis='y', labelsize=fontsize)
            if pred_corr_lbl:
                axes.set_ylabel('Prediction correlation', fontsize=fontsize)
            
        if pd is False:
            if i == 2:
                axes.set_title(title, x=0.1,y= 1.04,loc='center', fontsize=fontsize)
        elif pd:
            if i == 1:
                axes.set_title(title, x=0.1,y= 1.05,loc='center', fontsize=fontsize)

            
        if i >= 1:
            axes.spines['left'].set_visible(False)
            axes.spines['right'].set_visible(False)
            axes.tick_params(axis='y', length=0)
            axes.set_yticks([])
        
    plt.subplots_adjust(wspace=0)

def plot_corr(pred_corr1, pred_corr2, title1, title2, xup_lim=1, fontsize=13):
    masked_pred_corr_1 = np.ma.masked_invalid(pred_corr1) 
    masked_pred_corr_2 = np.ma.masked_invalid(pred_corr2) 
    corr_coef = np.ma.corrcoef(masked_pred_corr_1, masked_pred_corr_2)
    corr_value = corr_coef[0, 1]


    plt.scatter(pred_corr1,pred_corr2, color= 'none', edgecolor='black', alpha=0.5)  
   
    plt.axline(xy1=(0, 0), xy2=(0.5, 0.5), color='red', alpha=0.6, linestyle='-') 
    plt.legend([])
    plt.xlabel(title1, fontsize=fontsize)    
    plt.ylabel(title2, fontsize=fontsize)   
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylim(-0.15,xup_lim)
    plt.xlim(-0.15,xup_lim)
    plt.tick_params(axis='both', labelsize=fontsize)  

