package als;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
//import org.apache.spark.mllib.recommendation.ALS;
//import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;

import scala.Tuple2;

import org.apache.log4j.Logger;
import org.apache.log4j.Level;

//Jiaxi Li
/**
 * This is the Java version of the Movie recommendation tutorial in Spark Summit
 * 2014. The original tutorial can be found:
 * https://databricks-training.s3.amazonaws.com/movie-recommendation-with-mllib.
 * html
 *
 * The Python and Scala code can be downloaded from
 * https://databricks-training.s3.amazonaws.com/getting-started.html
 *
 * The Java version uses the latest movie data set collected by grouplens.org
 * http://grouplens.org/datasets/movielens/
 *
 * The file format is slightly different to the ones used in the SparkSumit
 * tutorial. The personalized movie recommendation part is implemented in a
 * slightly different way. It calls the recommendProducts method directly on the
 * best model to get the recommendation list. The tutorial solution code
 * computes all ratings for movies that the user has not rated and sort them to
 * make recommendation.
 *
 */

public class MovieLensALS {

	public static void main(String[] args) {

		String inputDataPath = args[0];
		String ratingFile = args[1];

		// turn off Spark logs
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
		SparkConf conf = new SparkConf();

		conf.setAppName("Movie Recommendation ALS");
		JavaSparkContext sc = new JavaSparkContext(conf);

		JavaRDD<Rating> myRatings = sc.parallelize(loadRating(ratingFile));

		// Create a coordinate matrix from the rating data -- row: movies, column: users
		JavaRDD<MatrixEntry> userEntries = sc.textFile(inputDataPath + "ratings.csv").map(line -> {
			String[] data = line.split(",");
			return new MatrixEntry(Long.parseLong(data[1]), Long.parseLong(data[0]), Double.parseDouble(data[2]));
		});
		
		JavaPairRDD<Long, String> movies = sc.textFile(inputDataPath + "movies.csv").mapToPair(line -> {
			String[] data = line.split(",");
			
			return new Tuple2<Long, String>(Long.parseLong(data[0]), data[1]);
		});
		
		CoordinateMatrix userMat = new CoordinateMatrix(userEntries.rdd());//.transpose();

		// Convert the rating data into RowMatrix
		IndexedRowMatrix indexedUserRatingMatrix = userMat.toIndexedRowMatrix();
		RowMatrix userRatingMatrix = indexedUserRatingMatrix.toRowMatrix();

		long userNumCols = userRatingMatrix.numCols(), userNumRows = userRatingMatrix.numRows();
		System.out.println("The user matrix has " + userNumRows + " rows and " + userNumCols + " columns");

		// Compute basic summary of movies
		System.out.println("Compute summary of the rating matrix...");
		MultivariateStatisticalSummary userSummary = userRatingMatrix.computeColumnSummaryStatistics();
		
		// Since the matrix is very sparse, the only useful statistics here is nonzeros
		// the number of none zeros indicate the popularity of movie
		System.out.println("Compute mean for each user...");
		double[] userNonzeros = userSummary.numNonzeros().toArray();
		double[] userMeans = userSummary.mean().toArray();

		// compute the mean of all user with the movie they rate
		int k = 0;
		for (double u : userNonzeros) {
			userMeans[k] = userMeans[k++] * userNumRows / u;
		}
		//System.out.println("MEAN: " + Arrays.toString(userMeans));
		// keep a copy of the personalized movies and ratings
		int[] myMovies = new int[15];
		double[] myRates = new double[15];
		k = 0;
		for (Rating r : myRatings.collect()) {
			myMovies[k] = r.product();
			myRates[k] = r.rating();
			k++;
		}	

		System.out.println("Compute similarities among 15 movies...");
		
		JavaRDD<IndexedRow> myMovieRows = indexedUserRatingMatrix.rows().toJavaRDD().filter(f -> {
			for(int m : myMovies)
				if(f.index() == m)
					return true;
			return false;	
		});
		
		k = 0;
		int len = myMovieRows.collect().size();
		JavaRDD<MatrixEntry> mySim = sc.emptyRDD();
		System.out.println(Arrays.toString(myMovies));
		for(IndexedRow row : myMovieRows.collect()){
			System.out.print("Progress: " + k++ + "/"  + len + "\r");
//			System.out.println(row.index());
//			System.out.flush();
			JavaRDD<MatrixEntry> tmpSimilarity = indexedUserRatingMatrix.rows().toJavaRDD().filter(f -> {
				
				for(int m : myMovies)
					if(f.index() == m)
						return false;
				return true;
			}).map(r -> {

				double[] movieI = r.vector().toArray();
				double[] movieJ = row.vector().toArray();
				
				double sim = computeSimilarity(movieI, movieJ, userMeans);
				return (new MatrixEntry((long)r.index(), (long)row.index(), (double)sim));
			});
			mySim = mySim.union(tmpSimilarity);
		}

		System.out.println("Covert into indexed similarity row matrix...");
		CoordinateMatrix mySimMat = new CoordinateMatrix(mySim.rdd());
		IndexedRowMatrix mySimMatrix = mySimMat.toIndexedRowMatrix();
		System.out.println("SIM_MATRIX. Rows: "+ mySimMatrix.numRows() + " Columns: " + mySimMatrix.numCols());
		for(IndexedRow s : mySimMatrix.rows().toJavaRDD().collect()){
            if(s.index()==78499)
                System.out.println(s.index() + ": " + s.vector());
        }

		System.out.println("Compute prediction for other movies...");
		JavaRDD<Tuple2<Double, Long>> sortPredictionTuple = mySimMatrix.rows().toJavaRDD().map(r -> {
			double[] tmpSim = new double[15];
			double[] sims = r.vector().toArray();
            int i = 0;
			for(int m : myMovies)
				tmpSim[i++] = sims[m];
			
			ArrayIndexComparator comparator = new ArrayIndexComparator(tmpSim);
			Integer[] indices = comparator.createIndexArray();
			Arrays.sort(indices, comparator);

			double sum_down = 0;
			double sum_up = 0;
			for(i = 0; i < 10; i++){
                if(tmpSim[indices[i]] == -2)
                    continue;
				sum_down += Math.abs(tmpSim[indices[i]]);
				sum_up += tmpSim[indices[i]] * myRates[indices[i]];
			}
            
			double pred = sum_up / sum_down;
            if(Double.isNaN(pred))
                pred = 0;
			return (new Tuple2<Double, Long>(pred, (r.index())));
		});
		
		System.out.println("Sort the predictions...");
		System.out.println("My ratings: " + Arrays.toString(myRates));

		JavaPairRDD<Double, Long> sortPredictionPair = JavaPairRDD.fromJavaRDD(sortPredictionTuple);
		List<Tuple2<Double, Long>> sortedResults = sortPredictionPair.sortByKey(false).collect();
		
		k = 0;
		System.out.println("Done! Output top 50...");
		List<Long> movieIndex = new ArrayList<Long>();
		List<Double> predictResults = new ArrayList<Double>();
		
		for(Tuple2<Double,Long> t : sortedResults){
			if(k++ >= 50)
				break;

			movieIndex.add(t._2);
			predictResults.add(t._1);
		}
		
		JavaPairRDD<Long, String> top50 = movies.filter(t ->{
			for(long i : movieIndex){
				if(i == t._1)
					return true;
			}
			return false;
		});
		
		Object[] movieArray = movieIndex.toArray();
		Object[] predictArray = predictResults.toArray();
		
		for(k = 0; k < movieIndex.size(); k++){
			String movieName = findMovieName((long)movieArray[k], top50);

			if (movieName != null)
				System.out.println(k + ". ->" + movieArray[k] + "<-" + movieName + "'s predicted rating is: " + (double)predictArray[k]);
		}
		
		System.exit(0);
	}
	
	
	// method for calculating the similarity between 2 movie ratings
	static double computeSimilarity(double[] movieI, double[] movieJ, double[] userMeans) {
		double res_up = 0;
		double res_downI = 0;
		double res_downJ = 0;
		double res_down = 0;
		for(int i = 1; i < movieI.length; i++){
		     if(userMeans[i] <= 0)
                continue;    
			 if (movieI[i] == 0 || movieJ[i] == 0)
                continue;
             res_up += (movieI[i] - userMeans[i]) * (movieJ[i] - userMeans[i]);
			 res_downI += (movieI[i] - userMeans[i]) * (movieI[i] - userMeans[i]);
			 res_downJ += (movieJ[i] - userMeans[i]) * (movieJ[i] - userMeans[i]);
		}
		
		res_down = Math.sqrt(res_downI) * Math.sqrt(res_downJ);
        if(Double.isNaN(res_up/res_down))
            return -2;

		return res_up / res_down;
	}
	
	// find movie name in the filtered movie list
	private static String findMovieName(long index, JavaPairRDD<Long, String> top50) {
		for(Tuple2<Long, String> t : top50.collect()){
			if (t._1 == index)
				return t._2;
		}
		System.out.println("not found:" + index);
		return null;
	}
	
	// method for loading personalized ratings  
	private static List<Rating> loadRating(String ratingFile) {

		ArrayList<Rating> ratings = new ArrayList<Rating>();
		try {
			BufferedReader reader = new BufferedReader(new FileReader(ratingFile));
			String line;
			while ((line = reader.readLine()) != null) {
				String[] data = line.split(",");
				ratings.add(
						new Rating(Integer.parseInt(data[0]), Integer.parseInt(data[1]), Double.parseDouble(data[2])));

			}
			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return ratings;

	}
}
