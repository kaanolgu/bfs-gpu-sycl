const char* datasetName;
#define SIZE_ONE 1

std::vector<unsigned int > source_meta(SIZE_ONE);
std::vector<unsigned int > source_indptr(SIZE_ONE);
std::vector<unsigned int > source_inds(SIZE_ONE);


std::vector<unsigned int > source_meta_cpu(SIZE_ONE);
std::vector<unsigned int > source_indptr_cpu(SIZE_ONE);
std::vector<unsigned int > source_inds_cpu(SIZE_ONE);

size_t result;

unsigned int getFileSize(const char *fileName) {
	  unsigned int fil_size;
	  FILE * pFile;
	  pFile = fopen ( fileName , "r" );
	  if(pFile == NULL){
		  std::cout << "ERROR (1) : File " << fileName << " not found! " << std::endl;
		  assert(1);

	  }
	  fseek(pFile, 0, SEEK_END); // seek to end of file
	  fil_size = ftell(pFile); // get current file pointer
	  if(fil_size == 0){
		  std::cout << "ERROR (2) : File size is 0 !" << std::endl;
	  }
	fclose(pFile);
	return fil_size;
}

void readIntoBuffer(const char * fileName,std::vector<unsigned int >&buffer, unsigned int bufsize,std::vector<unsigned int> &old_buffer_size,unsigned int &offset,int index) {
	  FILE * pFile;
	  pFile = fopen ( fileName , "r" );
	  fseek( pFile , 0 , SEEK_SET );
	//   std::cout << old_buffer_size[index] << std::endl;
	  result = fread(buffer.data() + offset,1,bufsize,pFile);
	  if(result != bufsize){
		  std::cout << " READ ERROR " << std::endl;
	  }
	  fclose ( pFile );
	  offset += bufsize/4;
	  old_buffer_size.push_back(offset);
	//    for(int i = 0; i < 30; i++){
		//    std::cout << "pointer[" << i << "] = "<< buffer[i] << std::endl;
	//    }

}

void readFromMM(const char * fileName,std::vector<unsigned int >&buffer,std::vector<unsigned int> &old_buffer_size,unsigned int &offset,int index) {
	std::cout << "Reading " << fileName << "..." ;
	unsigned int fileSize = getFileSize(fileName);
	// std::cout << "fileSize : " << fileSize << std::endl;
	// define glob_buffer_size to keep track of the last file size written to the buffer in order to make sure we continue where we left off
	buffer.resize(fileSize + old_buffer_size[index]);
	readIntoBuffer(fileName,buffer, fileSize,old_buffer_size,offset,index);
	std::cout << "OK" << std::endl;
}


void loadMatrix(unsigned int partitionCount,	std::vector<unsigned int>& old_buffer_size_meta,
	std::vector<unsigned int>& old_buffer_size_indptr,
	std::vector<unsigned int>& old_buffer_size_inds,unsigned int &offset_meta,unsigned int &offset_indptr,unsigned int &offset_inds) {
	// datasetName = rmat-20-32
	std::cout << "Loading matrix " << datasetName << " with " << partitionCount << " partitions.." << std::endl;  
	std::string pth = "/dataset/";
	std::string non_switch = getenv("PWD") + pth;
	std::string temp = datasetName;

	
	non_switch += temp + "-csc-" + std::to_string(partitionCount) + "/" + temp + "-csc-";


	unsigned int zeroz = 0;
	for(int i = 0; i < partitionCount; i++){
		std::string str_meta =non_switch +std::to_string(i)+ "-meta.bin";
		std::string str_indptr = non_switch +std::to_string(i) +"-indptr.bin";
		std::string str_inds = non_switch +std::to_string(i)+ "-inds.bin";
		
  		readFromMM(str_meta.c_str(), source_meta,old_buffer_size_meta,offset_meta,i);
		readFromMM(str_indptr.c_str(), source_indptr,old_buffer_size_indptr,offset_indptr,i);
		readFromMM(str_inds.c_str(), source_inds,old_buffer_size_inds,offset_inds,i);
		
	}

  // cache the loaded matrix details to reuse later





//  for(int i =old_buffer_siznds[partitionCount-1]/4-5 ; i < old_buffer_size_inds[partitionCount-1]/4+30; i++)
//          std::cout << "\nsource_inds[" << i << "] : " << std::setw(10) << std::left << source_inds[i];


}
void loadMatrixCPU(unsigned int partitionCount,	std::vector<unsigned int>& old_buffer_size_meta,
	std::vector<unsigned int>& old_buffer_size_indptr,
	std::vector<unsigned int>& old_buffer_size_inds,unsigned int &offset_meta,unsigned int &offset_indptr,unsigned int &offset_inds) {
	// datasetName = rmat-20-32
	std::cout << "Loading matrix " << datasetName << " with " << partitionCount << " partitions.." << std::endl;  
	std::string pth = "/dataset/";
	std::string non_switch = getenv("PWD") + pth;
	std::string temp = datasetName;

	
	non_switch += temp + "-csc-" + std::to_string(partitionCount) + "/" + temp + "-csc-";
	
	
	unsigned int zeroz = 0;
	for(int i = 0; i < partitionCount; i++){
		std::string str_meta =non_switch +std::to_string(i)+ "-meta.bin";
		std::string str_indptr = non_switch +std::to_string(i) +"-indptr.bin";
		std::string str_inds = non_switch +std::to_string(i)+ "-inds.bin";
		
  		readFromMM(str_meta.c_str(), source_meta_cpu,old_buffer_size_meta,offset_meta,i);
		readFromMM(str_indptr.c_str(), source_indptr_cpu,old_buffer_size_indptr,offset_indptr,i);
		readFromMM(str_inds.c_str(), source_inds_cpu,old_buffer_size_inds,offset_inds,i);
		
	}

  // cache the loaded matrix details to reuse later





//  for(int i =old_buffer_siznds[partitionCount-1]/4-5 ; i < old_buffer_size_inds[partitionCount-1]/4+30; i++)
//          std::cout << "\nsource_inds[" << i << "] : " << std::setw(10) << std::left << source_inds[i];


}
