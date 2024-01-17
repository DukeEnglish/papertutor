// script.js
document.addEventListener('DOMContentLoaded', function() {
    // 获取所有小标题的链接
    const links = document.querySelectorAll('#categories a');
    const sections = document.querySelectorAll('section');

    // 为每个链接添加点击事件监听器
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault(); // 防止链接跳转
            const targetId = this.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);

            // 切换当前激活的section
            sections.forEach(section => {
                section.style.display = section.id === targetSection.id ? 'block' : 'none';
            });
        });
    });
});