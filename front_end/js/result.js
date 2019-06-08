document.getElementById("single-query-btn").onclick = function(){
	var query_text = $("#single-query").val();
	if(input_check(query_text, function(){$("#onerror-modal").modal("show");})){
		$.ajax({
			url : '',
			type : 'POST',
			data : query_text,
	      	async: false,
			processData : false,
			contentType : false,
			beforeSend : function(){
				awaiting();
			}
		});
	}
}