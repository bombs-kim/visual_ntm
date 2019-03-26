var memory;
var memory_prev;
var read_head;
var write_head;


function update(response) {
    if (response) {
        memory = response.memory;
        memory_prev = response.memory_prev;
        read_head = response.read_head;
        write_head = response.write_head;
        var diff_only = false;
    }
    else {
        var diff_only = true;
    }
    // draw memory state
    draw_grid(60, 60, memory, memory_prev, 8, diff_only);
    if (diff_only){
        return;
    }

    draw_grid(400, 60, read_head, null, 10, diff_only);
    draw_grid(400, 200, write_head, null, 10, diff_only);
}


function draw_grid(x, y, mat, mat_prev, grid_size, diff_only){
    var gs = grid_size;
    if (!mat)
        return;

    for (j = 0; j < mat.length; j++) {
        for (i = 0; i < mat[j].length; i++){
            stroke(0);

            // The color is chosen somewhat arbitrarily here.
            var val = mat[j][i]*100;
            val = 1 / (1 + Math.pow(Math.E, -val));
            var color = val * 200;

            if (mat_prev){
                var diff = Math.abs(mat[j][i] - mat_prev[j][i]);
            }    
            else
                var diff = 0;

            // Ignore small diff
            if (diff_only && diff < 0.001){
                continue;
            }

            strokeWeight(1);
            fill(0, color, color);
            rect(x + gs * i, y + gs * j, gs, gs);

            // Blinking effect
            if (diff > 0.001) {
                var accelerator = 16;
                var cur = (frameCount * accelerator) % 256;
                if (cur <= 16 * 8)
                    var mul = cur;
                else
                    var mul = (16 * 16) - cur;

                strokeWeight(2);
                diff = diff * mul;
                stroke(diff, 0, 0);
                rect(x + gs * i, y + gs * j, gs, gs);
            }
        }
    }
}


function put_text(text, left, top){
    var style = {
        'position': 'absolute',
        'left': left,
        'top': top
    }
    $("<div><div>")
        .text(text)
        .css(style)
        .appendTo('html');
}


function setup() {
    background(250);
    createCanvas(800, 1000);
    frameRate(8);
    put_text('memory', 70, 45);
    put_text('read_head', 410, 45);
    put_text('write_head', 410, 185);
}


function draw() {
    if (frameCount % 4 == 0){
        $.get('./data.json', update);
    } else {
        update();
    }
}
