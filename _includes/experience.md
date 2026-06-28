<h2 id="research-experience">Research Experience</h2>

<div class="experiences">
  {% for experience in site.data.experience.main %}
  <div class="experience-row">
    <div class="experience-logo">
      {% if experience.logo %}
      <img src="{{ experience.logo | relative_url }}" alt="{{ experience.school }} logo">
      {% endif %}
    </div>
    <div class="experience-content">
      <div class="experience-school">{{ experience.school }}</div>
      <div class="experience-role">{{ experience.role }}</div>
      <div class="experience-advisor">Advisor: {{ experience.advisor }}</div>
      <div class="experience-lab">
        {% if experience.lab_url %}
        <a href="{{ experience.lab_url }}" target="_blank" rel="noopener">{{ experience.lab }}</a>
        {% else %}
        {{ experience.lab }}
        {% endif %}
      </div>
    </div>
    <div class="experience-meta">
      <div class="experience-location">{{ experience.location }}</div>
      <div class="experience-status">{{ experience.status }}</div>
    </div>
  </div>
  {% endfor %}
</div>
