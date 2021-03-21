library(ggplot2)

df <- read.csv('/mnt/3602F83B02F80223/Downloads/imperial/Sem 2 - Machine Learning/Project/Data/ureadmissions.csv')

ggplot(df, aes(x=DAYS_NEXT_UADM)) +
    geom_histogram() +
    scale_x_log10()

ggplot(df, aes(x=DAYS_NEXT_UADM)) +
    geom_histogram(binwidth = 30) +
    xlim(3365, 4500)
