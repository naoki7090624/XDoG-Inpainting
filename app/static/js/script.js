$(function(){

    $('#loading').hide();
    //$('#canvasButton').show();
    var canvas_img = new fabric.Canvas('cnvs_img', {preserveObjectStacking: true});
    var img;
    var burshWidth = 10;
    var model;
    var canvas_mask;
    console.log(document.getElementById('BrushSize').value)


    // Load image and apply mask
    $('#inImg').on('change', function (e) {
        $('#uploader').hide();
        $('.canvas').show();
        img = new Image();
        var reader = new FileReader();
        //console.log(model);

        //Set image on the canvas
        reader.onload = function (e) {
            canvas_img.clear();
            img.src = reader.result;
            fabric.Image.fromURL(this.result, function(oImg) {
                canvas_img.add(oImg);
            });
        };
        reader.readAsDataURL(e.target.files[0]);

        $('#cnvs_mask').show();
        canvas_mask = new fabric.Canvas('cnvs_mask', {preserveObjectStacking: true});
            canvas_mask.forEachObject(function(object){
            object.selectable = false;
        });
        $('.drawingmode').show();
        canvas_img.isDrawingMode = false;
        canvas_mask.isDrawingMode = true;
        canvas_mask.freeDrawingBrush.width = burshWidth;
        canvas_mask.freeDrawingBrush.color = "white";
    });


    // change brush size
    $('#BrushSize').on('change', function() {
        let val = parseInt(this.value, 10) || 1;
        $('.value').html(val);
        burshWidth = val;
        if (canvas_mask){
            canvas_mask.freeDrawingBrush.width = parseInt(this.value, 10) || 1;
            //canvas_mask.freeDrawingBrush.width = val;
        }
    });

    // Clear Mask
    $('#ClearMask').on('click', function(){
        canvas_mask.clear();
    });


    // Clear loaded image
    $('#changeImg').on('click', function(){
        location.reload();
        canvas_img.clear();
        canvas_mask.clear();
    });


    // Send image and mask to flask server
    $('#sendImg').on('click', function(){

        $('#loading').show();
        $('#canvasButton').hide();
        $('#canvasDescription').hide();
        var canvas_img = document.getElementById('cnvs_img');
        var canvas_mask = document.getElementById('cnvs_mask');

        //encode image and mask as base64
        var enc_img = canvas_img.toDataURL('image/png');
        var enc_mask = canvas_mask.toDataURL('image/png');

        //Set checkpoint (canny, DoG, XDoG)
        var ckpt = document.getElementById('select-ckpt')
        var database = document.getElementById('select-dataset')
        model = ckpt.options[ckpt.selectedIndex].value;
        dataset = database.options[database.selectedIndex].value;

        var JSONdata = {
            model,
            dataset,
            img: enc_img,
            mask: enc_mask
        };

        $.ajax({
            url: 'http://127.0.0.1:5000/send_img',
            type: 'POST',
            data : JSON.stringify(JSONdata),
            contentType: 'application/JSON',
            dataType : 'JSON',
            processData: false,

            success: function(data, dataType) {
                if (data.ResultSet.ip_type == 'inpaint_success') {
                     console.log('Success', data);
                     var image = document.getElementById("content-img");
                     //var masked = document.getElementById("content-masked");
                     var edge = document.getElementById("content-edge");
                     var result = document.getElementById("content-result");
                     image.src = data.ResultSet.img;
                     //masked.src = data.ResultSet.masked;
                     edge.src = data.ResultSet.edge;
                     result.src = data.ResultSet.result;

                    //$('#content-before').hide();
                    $('#loading').hide();
                    $('#content-after').show();

                 }
            },
            error: function(XMLHttpRequest, textStatus, errorThrown) {
                console.log('Error : ' + errorThrown);
            }
        })
    });
});