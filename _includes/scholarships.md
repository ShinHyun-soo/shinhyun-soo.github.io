<h2 id="scholarships">Scholarships</h2>

<div class="scholarships">
  {% for scholarship in site.data.scholarships.main %}
  <div class="scholarship-row">
    <div class="scholarship-title">{{ scholarship.title }}</div>
    <div class="scholarship-status">{{ scholarship.status }}</div>
    <div class="scholarship-meta">{{ scholarship.organization }}</div>
  </div>
  {% endfor %}
</div>
