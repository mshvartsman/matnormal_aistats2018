library(data.table)
library(ggplot2)
library(plyr)

result_folders <- c('/mnt/jukebox/pniintel/cohen/ms44/nips2017_data/raider',
 '/mnt/jukebox/pniintel/cohen/ms44/nips2017_data/raider/results_bkup')

res_files_complete <- dir(result_folders, full.names=T, pattern="*.csv")

allres <- llply(res_files_complete, fread)
allres <- rbindlist(allres)

names_from <- c("dpmnsrm_ecm", "dpmnsrm_ecme", "srm","dpsrm_ecm")
names_to <- c("DP-MNSRM (ECM)", "DP-MNSRM (ECME)", "SRM", "DP-SRM")

allres[,fancy_names:=mapvalues(model, names_from, names_to)]

p <- ggplot(allres, aes(x=factor(features), y=mean_acc, ymin=mean_acc-std_acc, ymax=mean_acc+std_acc, group=fancy_names, fill=fancy_names)) + geom_bar(stat="identity", position=position_dodge()) +
	geom_linerange(position=position_dodge(width=0.9), colour="gray") + theme_bw(base_size=10) + 
	geom_abline(intercept=1/7, slope = 0, linetype='dashed') + 
	scale_fill_brewer("Method", type = "qual") +
	labs(x="Features", y="Accuracy", title="Classification performance, Raider") 

ggsave(filename="./classification_raider.pdf", plot=p, width=4, height=5)