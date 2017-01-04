package ml;

//import java.util.ArrayList;
//import java.util.Collections;
//import java.util.Iterator;
//import java.util.LinkedList;
//import java.util.List;
import java.util.*;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

/**
 * This is a naive solution to find top 5 movies per genre It uses groupByKey to
 * group all movies beloing to each genre and Jave's Collections method to sort
 * the movies based on rating count.
 * 
 * 
 * input data : movies.csv (only 1.33MB)
 * 
 * format movieId,title,genres. sample data 1,Toy Story
 * (1995),Adventure|Animation|Children|Comedy|Fantasy 102604,
 * "Jeffrey Dahmer Files, The (2012)",Crime|Documentary
 * 
 * Genres are a pipe-separated list; Movie titles with comma is enclosed by a
 * pair of quotes.
 * 
 * ratings.csv (541.96MB)
 * 
 * format userId,movieId,rating,timestamp
 * 
 * sample data 1,253,3.0,900660748
 */
public class MLGenreTopMoviesNaive {

	public static void main(String[] args) {
		String inputDataPath = args[0];
		String outputDataPath = args[1];
		SparkConf conf = new SparkConf();

		conf.setAppName("Genre TOP5");
		JavaSparkContext sc = new JavaSparkContext(conf);
		JavaRDD<String> ratingData = sc.textFile(inputDataPath + "ratings.csv"),
						movieData = sc.textFile(inputDataPath + "movies.csv");
		
		// Return a new RDD that is reduced into numPartitions partitions
		ratingData = ratingData.coalesce(10,true);
		
		/******************************** ALL DATA SET *****************************************/
		Map<String, UserRatingCount> allUsers = new HashMap<String, UserRatingCount>();
		
		// produced uid, (count sum, rate sum)
		JavaPairRDD<String, Tuple2<Double, Integer>> userRatingAll = ratingData.mapToPair(v -> {
			String[] data = v.split(",");
			String uid = data[0];
			double rates = Double.parseDouble(data[2]);
			return new Tuple2<String, Tuple2<Double, Integer>>(uid , new Tuple2<Double, Integer>(rates, 1));
		}).reduceByKey((n1, n2) -> {
				int countSum = n1._2 + n2._2;
				double rateSum = n1._1 + n2._1;
				return new Tuple2<Double, Integer>(rateSum, countSum);
		});
		
		for(Tuple2<String, Tuple2<Double, Integer>> r : userRatingAll.collect()){
			allUsers.put(r._1, new UserRatingCount(r._1, r._2._2, r._2._1));
		}
		
		// map all genres
		JavaPairRDD<String, String> movieGenres = movieData.flatMapToPair(f ->{
			String[] data = f.split(",");
			String mid = data[0];
			List<Tuple2<String, String>> allGenres = new ArrayList<Tuple2<String, String>>();
			if(data.length >= 3){
				String[] genres = data[data.length - 1].split("\\|");
				for(String genre : genres){
					allGenres.add(new Tuple2<String, String>(mid, genre));
				}
				
			}
			
			return allGenres;
		});
		
		// just for join
		JavaPairRDD<String, Tuple2<String, Double>> midUserRatings = ratingData.mapToPair( v -> {
			String[] data = v.split(",");
			String mid = data[1];
			String uid = data[0];
			double rates = Double.parseDouble(data[2]);
			return new Tuple2<String, Tuple2<String, Double>>(mid, new Tuple2<String, Double>(uid, rates));
		});
		
		// mid, (genre, (uid, rating))
		JavaPairRDD<String, Tuple2<String, Tuple2<String, Double>>> jointResults = movieGenres.join(midUserRatings);
		
		// (uid \t genre) , (ratingSum, countSum)
		JavaPairRDD<String, Tuple2<Double, Integer>> userGenreRates = jointResults.values().mapToPair(v -> {
				String uid = v._2._1;
				String genre = v._1;
				double rating = v._2._2;
				return new Tuple2<String, Tuple2<Double, Integer>>(uid + "\t" + genre,
																	new Tuple2<Double, Integer>(rating, 1));
		}).reduceByKey((n1, n2) -> {
			double rateSum = n1._1 + n2._1;
			int countSum = n1._2 + n2._2;
			return new Tuple2<Double, Integer>(rateSum, countSum);
		});
		
		// genre, (uid \t ratingsum \t countsum)
		JavaPairRDD<String, String> genreRating  = userGenreRates.mapToPair(v -> {
			String[] data = v._1.split("\t");
			String uid = data[0];
			String genre = data[1];
			double rateSum = v._2._1;
			int countSum = v._2._2;
			return new Tuple2<String, String>(genre, uid + "\t" + rateSum + "\t" + countSum);
			
		});
		
		genreRating = genreRating.coalesce(2, true);
		
		// genre, {uid, raingsum, countsum}
		JavaPairRDD<String, List<String>> top5 = genreRating.aggregateByKey(
                new ArrayList<String>(),
                1, 
                (topList, details)-> {
                    int size = topList.size();
                    int i = 0;

                    while (i < size){
                        String[] data = details.split("\t");
                        String[] dataInList = topList.get(i).split("\t");
                        int countInData = Integer.parseInt(data[2]); // 2
                        int countInList = Integer.parseInt(dataInList[2]); // 2
                        if (countInData > countInList) {
                            topList.add(i, details);
                            if (size == 5)
                                topList.remove(size - 1);
                            break;
                        }
                        i++;
                    }
                    if (i < 5 && topList.isEmpty())
                        topList.add(i, details);
                    
                    return topList;
                },
                (topList1, topList2)->{
                    
                	// divide and conquer 
                    LinkedList<String> sorted = new LinkedList<String>();
                    int i = 0, j = 0, k = 0;
                    while (k < 5){
                        
                    	if (i >= topList1.size()){
                        
                    		if (j < topList2.size())
                    			sorted.add(k, topList2.get(j++));
                        
                        }else if (j >= topList2.size()){
                        	sorted.add(k, topList1.get(i++));
                            
                        }else{
                            String[] data1 = topList1.get(i).split("\t");
                            String[] data2 = topList2.get(j).split("\t");
                            if (Integer.parseInt(data1[2]) >= Integer.parseInt(data2[2])){
                            	sorted.add(k, topList1.get(i));
                                i++;                            
                            }else{
                            	sorted.add(k, topList2.get(j));
                                j++;    
                            }
                            
                        }
                        
                        k++;
                    }
                    return sorted;
                }
        );
		
		
		// display ALL in format: genre, uid::count(genre)::count(data set)::mean(genre)::mean(data set)
		JavaPairRDD<String, ArrayList<String>> results = top5.mapToPair(v -> {
            ArrayList<String> tmp = new ArrayList<String>(); 
            for(String s : v._2){
                String[] data = s.split("\t");
                double meanGenre = Double.parseDouble(data[1])/Double.parseDouble(data[2]);
//                int idx = allUsers.indexOf();
                String result = data[0] + "::" + data[2] + "::" + allUsers.get(data[0]).getRatingCount()
                + "::" + String.valueOf(meanGenre) + "::" + allUsers.get(data[0]).getMeanRating();
                tmp.add(result);
            }
            return new Tuple2<String, ArrayList<String>>(v._1, tmp);
        });
		
		results.saveAsTextFile(outputDataPath + "Workload1_output");		
		sc.close();
	}
}
