/*
 * @Descripttion: 
 * @Author: Duke 叶兀
 * @E-mail: ljyduke@gmail.com
 * @Date: 2024-01-18 02:06:01
 * @LastEditors: Duke 叶兀
 * @LastEditTime: 2024-01-22 01:10:08
 */
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
    // 获取所有解读按钮
    const interpretButtons = document.querySelectorAll('.interpret-button');

    // 为每个解读按钮添加点击事件监听器
    interpretButtons.forEach(button => {
        button.addEventListener('click', function() {
            // 获取当前按钮的ID
            const paperId = this.getAttribute('data-id');
            // 根据ID查找对应的解读内容
            const interpretationDiv = document.getElementById('interpretation-' + paperId);
            // 切换显示状态
            interpretationDiv.style.display = interpretationDiv.style.display === 'none' ? 'block' : 'none';
        });
    });
});