function input_check(query_data, onerror_function){
	if(query_data.length <= 0){
		onerror_function();
		return false;
	}else{
		return true;
	}
}