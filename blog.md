---
layout: page
title: Blog
permalink: /blog/
---

Welcome to my blog! Here you'll find my thoughts on technology, programming, and other topics that interest me.

I will also post my notes on the paper here.

## Latest Posts

{% if site.posts.size > 0 %}
<ul class="post-list">
  {% for post in site.posts limit:3 %}
  <li>
    {% assign date_format = site.minima.date_format | default: "%b %-d, %Y" %}
    <span class="post-meta">{{ post.date | date: date_format }}</span>
    <h3>
      <a class="post-link" href="{{ post.url | relative_url }}">
        {{ post.title | escape }}
      </a>
    </h3>
    {% if site.show_excerpts %}
      {{ post.excerpt }}
    {% endif %}
  </li>
  {% endfor %}
</ul>
{% else %}
<p>No posts found.</p>
{% endif %}

## Sort by Categories

{% assign categories = site.categories | sort %}
{% if categories.size > 0 %}
  {% for category in categories %}
    {% assign posts = category[1] %}
<details>
  <summary><strong>{{ category[0] }}</strong> ({{ posts.size }})</summary>
  <ul style="margin-top: 10px;">
    {% for post in posts %}
    <li>
      <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    </li>
    {% endfor %}
  </ul>
</details>
  {% endfor %}
{% else %}
<p>No categories found.</p>
{% endif %}
