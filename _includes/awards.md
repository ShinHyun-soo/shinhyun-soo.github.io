<h2 id="awards">Awards</h2>

<div class="awards">
  {% for award in site.data.awards.main %}
  <div class="award-row">
    <div class="award-content">
      <div class="award-title">{{ award.title }}</div>
      <div class="award-challenge">{{ award.challenge }}</div>
      <div class="award-meta">{{ award.issuer }} &middot; {{ award.date }}</div>
    </div>
  </div>
  {% endfor %}
</div>
