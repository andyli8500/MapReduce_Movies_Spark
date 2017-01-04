package ml;

import java.io.Serializable;

public class UserRatingCount implements Comparable, Serializable{
	String uid;
	int ratingCount;
	double ratingSum;
	double meanRating;
	
	public UserRatingCount(String uid, int ratingCount, double ratingSum){
		this.uid = uid;
		this.ratingCount = ratingCount;
		this.ratingSum = ratingSum;
		this.meanRating = ratingSum / ratingCount;
	}
	
	public String getUser() {
		return uid;
	}
	
	public int getRatingCount() {
		return ratingCount;
	}
	
	public double getRatingSum() {
		return ratingSum;
	}
	
	public double getMeanRating() {
		return meanRating;
	}
	
	
	
	public String toString(){
		return uid + ":" + ratingCount + ", "+  meanRating;
	}
	
	@Override
	public int compareTo(Object o2) {
		// TODO Auto-generated method stub
		UserRatingCount ur = (UserRatingCount) o2;
        if (ratingCount < ur.ratingCount)
                return 1;
        if (ratingCount > ur.ratingCount)
                return -1;
        return 0;

	}
	

}
