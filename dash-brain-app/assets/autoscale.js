// 페이지 로드 후 모든 Plotly 그래프에 autoscale 적용
window.onload = function() {
    // DOM이 완전히 로드된 후 1초 기다린 후 실행
    setTimeout(function() {
        console.log("자동 스케일 적용 시도...");
        // 모든 Plotly 그래프 찾기
        var graphDivs = document.querySelectorAll('.js-plotly-plot');
        console.log(graphDivs.length + "개의 그래프 발견");
        
        // 각 그래프에 autoscale 적용
        for(var i=0; i < graphDivs.length; i++) {
            if(graphDivs[i]._fullLayout) {
                console.log("그래프 #" + i + "에 autoscale 적용");
                Plotly.relayout(graphDivs[i], {
                    'xaxis.autorange': true,
                    'yaxis.autorange': true
                });
            }
        }
    }, 1000);
};

// 이미지 드롭다운 변경 시 autoscale 적용
$(document).ready(function() {
    $('#image-dropdown').on('change', function() {
        // 이미지 로드 후 약간 지연
        setTimeout(function() {
            // 모든 그래프에 autoscale 적용
            var graphDivs = document.querySelectorAll('.js-plotly-plot');
            for(var i=0; i < graphDivs.length; i++) {
                if(graphDivs[i]._fullLayout) {
                    Plotly.relayout(graphDivs[i], {
                        'xaxis.autorange': true,
                        'yaxis.autorange': true
                    });
                }
            }
        }, 500);
    });
}); 