#spark-submit --class als.MovieLensALS --master spark://soit-hdp-pro-1.ucc.usyd.edu.au:7077  --total-executor-cores 10 SparkAdv.jar hdfs://soit-hdp-pro-1.ucc.usyd.edu.au:8020/share/movie/  personalRatings.txt

spark-submit --class als.MovieLensALS --master yarn-client  --num-executors 15 SparkAdv.jar hdfs://soit-hdp-pro-1.ucc.usyd.edu.au:8020/share/movie/demo/ personalRatingsAction.txt
