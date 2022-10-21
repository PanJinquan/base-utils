__kernel void helloworld(__global char *pIn, __global char *pOut)
{
	int iNum = get_global_id(0);
	pOut[iNum] = pIn[iNum] + 1;
}