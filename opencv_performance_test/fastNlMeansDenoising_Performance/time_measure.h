/**
	Time structs for easy time-interval measuring.
	
 Author: Savvas Sampaziotis
*/

#include <sys/time.h>

typedef struct timIntervalstruct
{
	struct timeval startwtime, endwtime;
	double seqTime;
} TimeInterval;


void tic(TimeInterval* timeInterval)
{
	gettimeofday (&(timeInterval->startwtime), NULL);
}


double toc(TimeInterval* timeInterval)
{
	gettimeofday (&(timeInterval->endwtime), NULL);
	timeInterval->seqTime = (double)((timeInterval->endwtime.tv_usec - timeInterval->startwtime.tv_usec)/1.0e6
			      + timeInterval->endwtime.tv_sec - timeInterval->startwtime.tv_sec);
	return timeInterval->seqTime;
}
