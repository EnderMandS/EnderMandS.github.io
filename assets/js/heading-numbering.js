(function() {
  function alreadyNumbered(text) {
    return /^\d+(\.\d+)*\s/.test(text.trim());
  }

  function numberArticle(article) {
    if (article.dataset.numbered === '1') return;
    const content = article.querySelector('.post-content, .entry-content, .content') || article;
    const headings = Array.from(content.querySelectorAll('h1,h2,h3,h4,h5,h6'))
      .filter(h => !h.classList.contains('no-number'));

    if (!headings.length) return;

    // Determine base level (often h2 in minima posts because h1 is the title outside .post-content)
    const levels = headings.map(h => parseInt(h.tagName.substring(1), 10));
    let baseLevel = Math.min.apply(null, levels);

    const counters = [0,0,0,0,0,0];

    headings.forEach(h => {
      const level = parseInt(h.tagName.substring(1), 10);
      const idx = level - baseLevel;
      if (idx < 0 || idx > 5) return;

      // Increment this level, reset deeper levels
      counters[idx] += 1;
      for (let i = idx + 1; i < counters.length; i++) counters[i] = 0;

      // Build number sequence (stop at last non-zero)
      const parts = [];
      for (let i = 0; i <= idx; i++) {
        if (counters[i] === 0) break;
        parts.push(counters[i]);
      }
      if (!parts.length) return;

      if (alreadyNumbered(h.textContent)) return;

      h.innerHTML = '<span class="section-number">' + parts.join('.') + '</span> ' + h.innerHTML;
      if (!h.id) {
        h.id = 'sec-' + parts.join('-');
      }
    });

    article.dataset.numbered = '1';
  }

  function run() {
    if (!/\/posts\//.test(window.location.pathname)) return;
    const articles = document.querySelectorAll('article');
    articles.forEach(numberArticle);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', run);
  } else {
    run();
  }
})();
