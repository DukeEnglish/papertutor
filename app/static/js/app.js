$(document).ready(function() {
    var currentPage = 1;
    var itemsPerPage = 10;
  
    // 获取论文数据并更新表格
    function updatePaperTable() {
      $.ajax({
        url: '/papers',
        type: 'GET',
        data: {
          page: currentPage,
          per_page: itemsPerPage
        },
        success: function(data) {
          // 清空表格内容
          $('#paper-list').empty();
  
          // 遍历论文数据并添加到表格
          for (var i = 0; i < data.length; i++) {
            var paper = data[i];
            var row = $('<tr>');
            row.append($('<td>').text(paper.title));
            row.append($('<td>').text(paper.authors));
            row.append($('<td>').html('<a href="' + paper.links + '">Link</a>'));
            row.append($('<td>').text(paper.entry_id));
            row.append($('<td>').html('<a href="' + paper.pdf_url + '">PDF</a>'));
            row.append($('<td>').text(paper.summary));
            row.append($('<td>').text(paper.updated));
            $('#paper-list').append(row);
          }
        }
      });
    }
  
    // 更新翻页按钮
    function updatePagination() {
      $.ajax({
        url: '/total_pages',
        type: 'GET',
        success: function(totalPages) {
          // 清空翻页按钮
          $('#pagination').empty();
  
          // 添加翻页按钮
          for (var i = 1; i <= totalPages; i++) {
            var button = $('<button>');
            button.text(i);
            button.click(function() {
              currentPage = parseInt($(this).text());
              updatePaperTable();
              updatePagination();
            });
            $('#pagination').append(button);
          }
        }
      });
    }
  
    // 初始化页面
    updatePaperTable();
    updatePagination();
  });