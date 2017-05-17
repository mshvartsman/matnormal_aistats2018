library(data.table)
library(ggplot2)
library(plyr)

result_folder_raider <- c('/mnt/jukebox/pniintel/cohen/ms44/nips2017_data/raider/results')
result_folder_sherlock <- c('/mnt/jukebox/pniintel/cohen/ms44/nips2017_data/sherlock/results')

res_files_raider <- dir(result_folder_raider, full.names=T, pattern="results*")
res_files_sherlock <- dir(result_folder_sherlock, full.names=T, pattern="results*")

res_raider <- rbindlist(llply(res_files_raider, fread))
res_raider[,chance_level:=1/7]
res_sherlock <- rbindlist(llply(res_files_sherlock, fread))
res_sherlock[,chance_level:=1/49]
res_raider[,experiment:="raider"]
res_sherlock[,experiment:="sherlock"]
allres <- rbind(res_raider, res_sherlock)

names_from <- c("dpmnsrm_ecm",
				"dpmnsrm_ecme",
				"dpmnsrm_orthos_ecm",
				"dpmnsrm_orthos_ecme",
				"dpsrm_ecm",
				"dpsrm_orthos_ecm",
				"srm")
names_to <- c("DP-MNSRM (ECM)",
			 "DP-MNSRM (ECME)",
			 "DP-MNSRM (ECM, Orth. S)",
			 "DP-MNSRM (ECME, Orth. S)",
			 "DP-SRM",
			 "DP-SRM (Ortho. S)",
			  "SRM")

allres[,fancy_names:=mapvalues(model, names_from, names_to)]

p <- ggplot(allres, aes(x=factor(features), y=mean_acc, ymin=mean_acc-std_acc, ymax=mean_acc+std_acc, group=fancy_names, fill=fancy_names)) + geom_bar(stat="identity", position=position_dodge()) +
	geom_linerange(position=position_dodge(width=0.9), colour="gray") + theme_bw(base_size=10) + 
	geom_hline(aes(yintercept=chance_level)) + 
	scale_fill_brewer("Method", type = "qual") + facet_wrap(~experiment, scales="free")
	labs(x="Features", y="Accuracy", title="Classification performance") 

ggsave(filename="./classification_raider.pdf", plot=p, width=4, height=5)