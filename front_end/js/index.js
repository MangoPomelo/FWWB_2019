document.getElementById("drag-and-drop-zone").addEventListener("dragover", function(event) {
    event.preventDefault();
}, false);
document.getElementById("drag-and-drop-zone").addEventListener("drop", function(event) {
	event.preventDefault();
	upload_file(event.dataTransfer.files[0]);
}, false);
document.getElementById("file").onchange = function() {
	upload_file(document.getElementById("file").files[0]);
}
document.getElementsByClassName("query-btn")[0].onclick = function(){
	var query_text = $("#text-query").val();
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

function awaiting(){
	$("#classify-by-text").hide();
	$("#classify-by-file").hide();
	$("#awaiting-pane").show();
}
function upload_file(file) {
	var form_data = new FormData();
	form_data.append('file', file);
	$.ajax({
		url : '',
		type : 'POST',
		data : form_data,
      	async: false,
		processData : false,
		contentType : false,
		beforeSend : function(){
			awaiting();	
		}
	});
}